# NPVI

In 2012, restrictive distributional assumptions were still a downside of Variational Inference (VI). These restrictions engulfed both variational distributions and model families in which inference was possible. In this adverse scenario, Gershman et. Al.\[[1]\]  proposed Non-parametric VI (NPVI), an approach for flexible inference on non-conjugate models using mixtures of diagonal Gaussians with identical component weights as variational distributions and using a block optimization scheme to make inference possible.

Some eight years latter, automatic differentiation toolboxes (PyTorch and Tensorflow) are everywhere. 
With these tools, we could do the same with much more ease simply using the reparameterization trick and a few samples from the variational distribution.  

Here, you can find a modern implementation of NPVI using PyTorch.

As an illutration, I've implemented it for inference on a non-conjugate hierarchical logistic regression model.

[1]: http://gershmanlab.webfactional.com/pubs/GershmanHoffmanBlei12.pdf
