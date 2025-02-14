import numpy as np


def save_to_txt(name: str, path: str, test_texts: list, test_set_masks: list, test_real_masks: list,
                predicted_classes: list, real_classes: list) -> None:

    assert len(test_texts) == len(test_set_masks) == len(test_real_masks) == len(predicted_classes) == len(real_classes)

    with open(path.format(name), "w", encoding="utf-8") as file:
        for i, (text, mask, real_masks, y_pred, y_true) in enumerate(zip(test_texts,
                                                                         test_set_masks,
                                                                         test_real_masks,
                                                                         predicted_classes,
                                                                         real_classes)):
            mask = mask[0: len(text)]
            file.write("Text n.{}:".format(i + 1) + "\n")
            original_string = ""
            masked_string = ""

            for word, val in zip(text, mask):
                original_string += word + " "
                if val:
                    masked_string += word + " "

            ann_strings: list[str] = []
            for real_mask in real_masks:
                ann_string = ""
                for word, val in zip(text, real_mask):
                    if val:
                        ann_string += word + " "
                ann_strings.append(ann_string)

            file.write("Original:    " + original_string + "\n")
            file.write("Masked:      " + masked_string + "\n")
            for j, ann_string in enumerate(ann_strings):
                file.write("True mask {}: ".format(j + 1) + ann_string + "\n")
            file.write("Pred. label: {}".format(np.argmax(y_pred)) + "\n")
            file.write("Real label:  {}".format(np.argmax(y_true)) + "\n")
            file.write("\n")


def save_metrics(name: str, path: str, metrics: dict) -> None:

    with open(path.format(name), "w", encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(str(key) + ": " + str(value) + "\n")
