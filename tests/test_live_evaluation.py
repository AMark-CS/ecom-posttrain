import unittest
from ecom_posttrain.deployment.live_evaluation import task_metrics


class LiveEvaluationTests(unittest.TestCase):
    def test_named_metrics_do_not_mix_accuracy_and_normalized_accuracy(self):
        report={'results':{'piqa':{'acc,none':.5,'acc_norm,none':.75,'acc_stderr,none':'N/A'}},
                'n-samples':{'piqa':{'original':1838,'effective':4}},'versions':{'piqa':1}}
        row=task_metrics(report,'piqa')
        self.assertEqual(row['acc'],.5)
        self.assertEqual(row['acc_norm'],.75)
        self.assertIsNone(row['acc_stderr'])
        self.assertEqual(row['samples']['effective'],4)
        self.assertEqual(row['version'],1)
        report['results']['piqa']['acc_stderr,none']=float('nan')
        self.assertIsNone(task_metrics(report,'piqa')['acc_stderr'])
