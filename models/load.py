from models.as_module import ActivationShapingModule
from models.resnet import BaseResNet18
from models.ras_resnet import RASResNet18
from models.da_resnet import DAResNet18

from globals import CONFIG


def load_model(experiment):
    if experiment in ["random_maps", "domain_adapt"]:
        shaping_module = ActivationShapingModule(
            topK=CONFIG.topK,
            topK_treshold=CONFIG.tk_treshold,
            binarize=not CONFIG.no_binarize,
            epsilon=CONFIG.epsilon,
        )

    if experiment == "random_maps":
        return RASResNet18(
            mask_ratio=CONFIG.mask_ratio,
            shaping_module=shaping_module,
            random_shape_layers=CONFIG.layers,
            use_bernoulli=CONFIG.use_bernoulli,
        )

    if experiment == "domain_adapt":
        return DAResNet18(shaping_module=shaping_module, adapt_layers=CONFIG.layers)

    return BaseResNet18()
