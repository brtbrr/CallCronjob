from enum import Enum, unique


@unique
class BertLabel(Enum):
    ***** = "*****"
    ***** = "*****"
    ATM_WITHDRAWALS = "банкоматы снятия"

BERT_LABELS = [x.name for x in list(BertLabel)]
