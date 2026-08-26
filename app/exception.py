class MyException(Exception):
    def __init__(self, arg=""):
        self.arg = arg

class InvalidDataShapeException(MyException):
    def __str__(self):
        return (f"{self.arg}")

class InvalidNodeSettingException(MyException):
    def __str__(self):
        return (f"{self.arg}")
    
class FunctionUndefinedException(MyException):
    def __str__(self):
        return (f"{self.arg}")

class NonDataException(MyException):
    def __str__(self):
        return ("データのないノードが作成されています.")

class InvalidArgumentException(MyException):
    def __str__(self):
        return (f"{self.arg}")