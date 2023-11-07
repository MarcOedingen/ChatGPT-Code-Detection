import numpy as np
import pandas as pd
import Utility.utils as utils


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    unique_task_ids = code_data["task_id"].unique()
    ai_rows = code_data[code_data["is_gpt"]]
    human_rows = code_data[~code_data["is_gpt"]]
    min_samples = np.zeros(len(unique_task_ids), dtype=np.int32)
    for i, task_id in enumerate(unique_task_ids):
        min_samples[i] = min(
            len(ai_rows[ai_rows["task_id"] == task_id]),
            len(human_rows[human_rows["task_id"] == task_id]),
        )
    ai_samples, human_samples = [], []
    for i, task_id in enumerate(unique_task_ids):
        ai_samples.append(
            ai_rows[ai_rows["task_id"] == task_id].sample(min_samples[i], replace=False)
        )
        human_samples.append(
            human_rows[human_rows["task_id"] == task_id].sample(
                min_samples[i], replace=False
            )
        )
    paired_samples = pd.concat(ai_samples + human_samples)
    utils.save_data(paired_samples, "Final_Datasets/Paired_Embedded_Cleaned")


main()
