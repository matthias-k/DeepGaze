# DeepGaze II and DeepGaze IIE

This repository contains the pytorch implementations of DeepGaze II and DeepGaze IIE

This is how use the pretained DeepGaze IIE model:

```python
from scipy.misc import face
from scipy.ndimage import zoom
import torch

DEVICE = 'cuda'

model = deepgaze_pytorch.deepgaze2e(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load('deepgaze2e.pth'))

image = face()

# load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
centerbias_template = np.load('centerbias.npy')
# rescale to match image size
centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
# renormalize log density
centerbias -= logsumexp(centerbias)

image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

log_density_prediction = model(image_tensor, centerbias_tensor))
```

If you use these models, please cite the according papers:

* DeepGaze II: [Kümmerer, M., Wallis, T. S. A., Gatys, L. A., & Bethge, M. (2017). Understanding Low- and High-Level Contributions to Fixation Prediction. 4789–4798.](http://openaccess.thecvf.com/content_iccv_2017/html/Kummerer_Understanding_Low-_and_ICCV_2017_paper.html)
* DeepGaze IIE: [Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. ArXiv:2105.12441 [Cs]](http://arxiv.org/abs/2105.12441)
