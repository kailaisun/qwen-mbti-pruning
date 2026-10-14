import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze_pruning_sweep import jsd,balanced_score
from mbti_common import load_items,score_mbti


class AnalysisTests(unittest.TestCase):
    def test_jsd_controls(self):
        p=np.array([[.1,.2,.7],[1.,0.,0.]])
        q=np.array([[.3,.6,.1],[0.,1.,0.]])
        np.testing.assert_allclose(jsd(p,p),0,atol=1e-12)
        np.testing.assert_allclose(jsd(p,q),jsd(q,p),atol=1e-12)
        self.assertTrue(np.all(jsd(p,q)>=0))
        self.assertTrue(np.all(jsd(p,q)<=np.log(2)+1e-12))
        self.assertAlmostEqual(jsd(p,q)[1],np.log(2))

    def test_constant_answer_controls(self):
        items=load_items()
        for option in 'ABCDEFG':
            choices=[option]*len(items)
            self.assertEqual(balanced_score(items,choices)[0],'????')
            expected='ENFJ' if option in 'ABC' else 'ISTP' if option in 'EFG' else '????'
            self.assertEqual(score_mbti(items,choices)[0],expected)


if __name__=='__main__':unittest.main()
