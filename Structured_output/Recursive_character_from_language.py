

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language

text= """
# Define a simple class
class Car:
    def __init__(self, brand, model, year):
        self.brand = brand  # Instance variable
        self.model = model
        self.year = year

    def start_engine(self):
        print(f"{self.brand} {self.model}'s engine started!")

    def stop_engine(self):
        print(f"{self.brand} {self.model}'s engine stopped.")

    def get_info(self):
        return f"{self.year} {self.brand} {self.model}"


# Create an object (instance) of Car
my_car = Car("Tesla", "Model S", 2024)

# Accessing methods
print(my_car.get_info())    # Output: 2024 Tesla Model S
my_car.start_engine()       # Output: Tesla Model S's engine started!
my_car.stop_engine()        # Output: Tesla Model S's engine stopped.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 100,
    chunk_overlap= 0
)

chunks = splitter.split_text(text)

print(chunks)