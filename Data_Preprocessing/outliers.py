import Utility.utils as utils


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    code_data["len_code"] = code_data["code"].apply(lambda x: len(x))
    normalized_len_code = (
        code_data["len_code"] - code_data["len_code"].mean()
    ) / code_data["len_code"].std()
    code_data = code_data[(normalized_len_code > -3) & (normalized_len_code < 3)]
    utils.save_data(code_data, "Final_Datasets/Paired_Embedded_Cleaned")


main()
