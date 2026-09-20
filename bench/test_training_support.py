import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch
from PIL import Image
from training_support import PublicationAugment, validation_panel


def load(name, filename):
    spec=importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class TrainingChecks(unittest.TestCase):
    def test_resume_batches_continue_instead_of_repeat(self):
        m=load('training','50_finetune.py')
        groups={str(i):list(range(i*4,i*4+4)) for i in range(8)}
        a=SimpleNamespace(seed=11,P=4,K=2,steps_b=10)
        whole=m.ListBatches(groups,list(groups),a,0)
        resumed=m.ListBatches(groups,list(groups),a,6)
        self.assertEqual(whole.batches[6:],resumed.batches)
        for x,y in zip(whole.labels[6:],resumed.labels): np.testing.assert_array_equal(x,y)

    def test_validation_panel_disjoint_and_order_independent(self):
        rows=[dict(pdb_l=p,view=v,split=s,render_id=f'{p}_{v}')
              for p,s in [('a','val'),('b','val'),('c','train'),('d','test')] for v in range(4)]
        df=pd.DataFrame(rows)
        a,h=validation_panel(df,1,9)
        b,h2=validation_panel(df.sample(frac=1),1,9)
        self.assertEqual(h,h2);self.assertTrue((a.split=='val').all())
        self.assertFalse(set(a[a.view==0].render_id)&set(a[a.view>0].render_id))

    def test_augment_dimensions_finite_and_seeded(self):
        aug=PublicationAugment(224)
        im=Image.new('RGB',(320,160),'white')
        for seed in range(20):
            torch.manual_seed(seed); x=np.asarray(aug(im))
            torch.manual_seed(seed); y=np.asarray(aug(im))
            self.assertEqual(x.shape,(224,224,3));np.testing.assert_array_equal(x,y)
            self.assertTrue(np.isfinite(x).all())

    def test_aggregation_variable_view_counts(self):
        import retrieval_scoring as m
        groups=torch.tensor([0,0,1,2,2,2])
        idx,mask=m.build_pad_index(groups,3,'cpu')
        sim=torch.tensor([[.9,.5,.7,.1,.8,.3],[-.2,-.4,-.9,-.7,-.6,-.8]])
        for rule,k in [('max',1),('top3',3),('top5',5)]:
            actual=m.aggregate(sim,idx,mask,rule)
            expected=torch.stack([torch.stack([row[groups==g].topk(min(k,int((groups==g).sum()))).values.mean()
                                      for g in range(3)]) for row in sim])
            torch.testing.assert_close(actual,expected)


if __name__=='__main__': unittest.main()
