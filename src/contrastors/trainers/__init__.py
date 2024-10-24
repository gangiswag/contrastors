from .base import *
# from .glue import *
# from .image_text import *
# from .mlm import *
# from .text_text import *
from .query_document import *

TRAINER_REGISTRY = {
    "encoder": TextTextTrainer,
    "query_document": QueryDocumentTrainer
}
