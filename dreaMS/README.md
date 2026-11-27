# Pretrained dreaMS backbone

`config.DREAMS_CKPT_DIR` points here. Place the dreaMS self-supervised
checkpoint at:

```
dreaMS/ssl_model.ckpt
```

It is about 1.39 GB and is **not** redistributed in this repository. Download it
from the official dreaMS release (see the dreaMS GitHub repository,
<https://github.com/pluskal-lab/dreaMS>) and keep the original file name.

`model_dreaMS.py` loads the Lightning checkpoint with a small custom unpickler,
so the `msml` package is not required.

If the file is missing, training prints a warning and continues from random
initialization. That is not the reported configuration and will not reproduce
the paper numbers.
