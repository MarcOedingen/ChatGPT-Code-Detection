import Utility.utils as utils


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    code_data = code_data.dropna()
    utils.save_data(code_data, "Final_Datasets/Paired_Embedded_Cleaned")


main()
