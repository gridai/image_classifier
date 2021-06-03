# image classifier
Image classification example!


```bash
pip install -r requirements.txt

python model.py --data_dir path/to/folder/with/train/val/test/split

```

## Data structure

Make sure your dataset has this structure:
```python
data/
  train/
    class_1/
      image.png
      image.png
    class_n/
      image.png
      image.png

  val/
    class_1/
      image.png
      image.png
    class_n/
      image.png
      image.png
      
  test/
    class_1/
      image.png
      image.png
    class_n/
      image.png
      image.png
```
