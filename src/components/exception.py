import sys
import logging

def error_msg(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Extract traceback info
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the filename
    errormsg = "Error encountered in the code [{0}] at line [{1}]  Error: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return errormsg


class CustomException(Exception):
    def __init__(self, errormsg, error_detail: sys):
        super().__init__(errormsg)
        self.errormsg = error_msg(errormsg, error_detail=error_detail)

    def __str__(self):
        return self.errormsg


# ✅ Main Execution Block
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Zero Division Error")
        raise CustomException(e, sys)
