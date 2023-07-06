import numpy as np
import torch

# a = np.random.randn(0, 2048)
a = torch.randn((0, 2048), dtype=torch.float32).numpy()
np.save("gallery.npy", a)

a = np.array(['person_1'])
a = np.array([], dtype=a.dtype)
np.save("names.npy", a)
