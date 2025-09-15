from typing import TypedDict

class Person(TypedDict):
    name: str
    age : int

new_Person: Person ={'name':'lucky','age':54}

print(new_Person)