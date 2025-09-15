
from typing import Optional
from pydantic import BaseModel,EmailStr,Field

class Student(BaseModel):
    name : str
    #default value
    frnd : str='abc'
    #may optional set
    age : Optional[int]=None
    # If must be in email format
    email : EmailStr
    # Field for checking greater or smaller
    score:  float = Field(gt=0,lt=100)

new_stu = {'name':"lucky",'email':'abc@gmail.com'}
obj = Student(**new_stu)

print(obj)