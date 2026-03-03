import os
import random
import numpy as np
import torch


def seed_all(seed: int = 1234) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map used throughout the project
LABEL_CLASSES = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
