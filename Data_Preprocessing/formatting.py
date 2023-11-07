import os
import subprocess
import Utility.utils as utils


def _format_code(code):
    with open("tmp.py", "w") as f:
        f.write(code)
    subprocess.run(["black", "tmp.py", "--fast"])
    with open("tmp.py", "r") as f:
        formatted_code = f.read()
    os.remove("tmp.py")
    return formatted_code


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    formatted_code = code_data["code"].apply(_format_code)
    code_data["code"] = formatted_code
    utils.save_data(code_data, "Final_Datasets/Paired_Embedded_Cleaned_Formatted")


main()
