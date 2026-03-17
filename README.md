From script directory run

```
python make_maps.py 
python measure_cls.py --mode lognormal --mask fullsky --recompute True
python measure_cls_wb.py --mode lognormal --mask fullsky --recompute True
python measure_cls.py --mode lognormal --mask tr1 --recompute True
python measure_cls_wb.py --mode lognormal --mask tr1 --recompute True
python natural_unmxing.py --mode lognormal --mask tr1 --recompute True
python natural_unmixing_wb.py --mode lognormal --mask tr1 --recompute True
```
