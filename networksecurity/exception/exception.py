import sys
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        """
        Custom exception class for handling errors in the Network Security project.

        Args:
            error_message (str): The error message.
            error_details (sys): The sys module to extract exception details.
        """
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        """
        String representation of the exception.

        Returns:
            str: Formatted error message with file name, line number, and error details.
        """
        return "Error occurred in Python script name [{0}] at line number [{1}] with error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )


if __name__ == '__main__':
    try:
        logger.logging.info("Entering the try block")
        a = 1 / 0  # Intentional error to demonstrate exception handling
        print("This will not be printed", a)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
