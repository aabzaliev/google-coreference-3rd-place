import json
import os

import numpy as np
import pandas as pd
from scipy.special import softmax

from predict import load_gap_test_data


def make_submission(settings):
    test = load_gap_test_data(settings)

    predictions_dir = settings['PREDICTIONS_DIR']
    pred_folders = os.listdir(predictions_dir)

    all_pred = list()
    for folder in pred_folders:
        print(folder)
        for f in os.listdir(os.path.join(settings['PREDICTIONS_DIR'], folder)):
            pred = np.load(os.path.join(settings['PREDICTIONS_DIR'], folder, f))
            all_pred.append(pred)

    assert np.unique([len(i) for i in all_pred]).item() == 12359

    av_preds = np.mean(np.stack(all_pred), axis=0)

    # Create submission file
    df_sub = pd.DataFrame(softmax(av_preds, -1).clip(1e-2, 1 - 1e-2), columns=["A", "B", "NEITHER"])
    df_sub["ID"] = test.ID
    df_sub.to_csv(os.path.join(settings['SUBMISSIONS_DIR'], "sub.csv"), index=False)


if __name__ == "__main__":
    with open('settings.json') as f:
        settings = json.loads(f.read())

    make_submission(settings)
