import unittest
import torch
from ecom_posttrain.model.grpo_loss import grpo_loss

class GRPOLossTests(unittest.TestCase):
    def test_kl_is_nonnegative_and_changes_policy_gradient(self):
        policy=torch.tensor([[-.2,-.8,-.4]],requires_grad=True)
        reference=torch.tensor([[-.8,-.1,-.9]])
        mask=torch.tensor([[0.,1.,1.]])
        advantages=torch.tensor([.5])
        a,_=grpo_loss(policy,mask,reference,mask,advantages,0)
        b,metrics=grpo_loss(policy,mask,reference,mask,advantages,.5)
        ga=torch.autograd.grad(a,policy,retain_graph=True)[0]
        gb=torch.autograd.grad(b,policy)[0]
        self.assertGreaterEqual(metrics['kl'],0)
        self.assertFalse(torch.allclose(ga,gb))
        self.assertEqual(gb[0,0].item(),0)

    def test_on_policy_ratio_and_advantage_direction(self):
        policy=torch.tensor([[-1.,-2.]],requires_grad=True)
        mask=torch.ones_like(policy)
        loss,metrics=grpo_loss(policy,mask,policy.detach(),mask,torch.tensor([1.]),0)
        self.assertAlmostEqual(loss.item(),-1.)
        self.assertEqual(metrics['kl'],0)
        loss.backward()
        self.assertTrue((policy.grad < 0).all())

    def test_response_mask_identity_and_empty_sequences_are_rejected(self):
        p=torch.zeros(1,2,requires_grad=True)
        for mask,refmask in [(torch.zeros_like(p),torch.zeros_like(p)),(torch.ones_like(p),torch.zeros_like(p))]:
            with self.assertRaises(ValueError):
                grpo_loss(p,mask,p.detach(),refmask,torch.ones(1),.1)

    def test_explicit_old_policy_clips_without_reference_gradients(self):
        policy=torch.zeros(1,2,requires_grad=True)
        old=torch.full((1,2),-2.,requires_grad=True)
        reference=torch.zeros(1,2,requires_grad=True)
        loss,_=grpo_loss(policy,torch.ones_like(policy),reference,torch.ones_like(policy),torch.ones(1),.1,old_tok=old)
        self.assertAlmostEqual(loss.item(),-1.2,places=6)
        loss.backward()
        self.assertTrue(torch.equal(policy.grad,torch.zeros_like(policy)))
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)

class TrainerTests(unittest.TestCase):
    def make_trainer(self, folder, stop=None, callback=None):
        from types import SimpleNamespace
        from unittest.mock import patch
        from ecom_posttrain.model.grpo_train import GRPOTrainer
        from ecom_posttrain.model.grpo_config import GRPOConfig
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.logits=torch.nn.Parameter(torch.zeros(10))
                self.generation_config=SimpleNamespace(eos_token_id=9)
            def forward(self,input_ids,**kwargs):
                return SimpleNamespace(logits=self.logits.expand(input_ids.shape[0],input_ids.shape[1],-1))
            def generate(self,**kwargs):
                return torch.tensor([[7,1,2,9,9],[7,1,3,4,9]])
            def save_pretrained(self,path): pass
        class Tokenizer:
            pad_token_id=9
            def decode(self,ids,**kwargs): return str(ids)
            def save_pretrained(self,path): pass
        def build(trainer):
            trainer.device='cpu'; trainer.model=Model(); trainer.ref_model=Model().requires_grad_(False).eval()
            trainer.tokenizer=Tokenizer();trainer.optimizer=torch.optim.AdamW(trainer.model.parameters(),lr=.1,weight_decay=0)
            trainer.prompts=[{'prompt':'x','input_ids':[7,1],'meta':{}}];trainer.dataset=trainer.prompts
        cfg=GRPOConfig(use_lora=False,output_dir=folder,group_size=2,prompts_per_step=1,max_steps=2,
                       learning_rate=.1,max_length=32,max_prompt_length=16,max_new_tokens=4,save_every=2)
        with patch.object(GRPOTrainer,'_build',build):
            trainer=GRPOTrainer(cfg,on_step=callback,should_stop=stop)
        return trainer

    def test_exact_tokens_eos_mask_and_optimizer_updates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            seen=[];trainer=self.make_trainer(folder,callback=seen.append)
            phases=[];trainer.on_phase=phases.append
            seq,texts=trainer.rollout([7,1],2)
            self.assertEqual(seq[0],([7,1,2,9],[-100,-100,2,9]))
            self.assertEqual(seq[1][0],[7,1,3,4,9])
            trainer.score_group=lambda text,meta:[0.,1.]
            trainer.fit()
            self.assertEqual(trainer.completed_steps,2)
            self.assertEqual([m['step'] for m in seen],[1,2])
            self.assertTrue(any('sampling prompt' in message for message in phases))
            self.assertTrue(any('backpropagating rollout' in message for message in phases))
            self.assertGreater(trainer.model.logits[3].item(),trainer.model.logits[2].item())
            self.assertTrue(all(p.grad is None for p in trainer.ref_model.parameters()))

    def test_flat_rewards_are_zero_advantage_and_cancel_preserves_steps(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            seen=[];trainer=self.make_trainer(folder,stop=lambda:len(seen)>=1,callback=seen.append)
            trainer.score_group=lambda text,meta:[1.,1.]
            trainer.fit()
            self.assertEqual(trainer.completed_steps,1)
            self.assertEqual(seen[0]['flat_groups'],1)
            self.assertTrue(torch.equal(trainer.model.logits,torch.zeros(10)))

    def test_log_probabilities_use_rollout_temperature(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            trainer=self.make_trainer(folder)
            with torch.no_grad(): trainer.model.logits.copy_(torch.arange(10.))
            trainer.cfg.temperature=2.
            ids=torch.tensor([[7,1,3]])
            labels=torch.tensor([[-100,-100,3]])
            logps,mask=trainer._token_logps(trainer.model,ids,labels,torch.ones_like(ids))
            expected=torch.log_softmax(torch.arange(10.)/2,-1)[3]
            self.assertAlmostEqual(logps[0,1].item(),expected.item())
            self.assertEqual(mask.tolist(),[[0.,1.]])

if __name__=='__main__': unittest.main()
