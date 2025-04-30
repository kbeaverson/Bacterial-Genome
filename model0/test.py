import datetime
import time

x = datetime.datetime.now()
time.sleep(5)
y = datetime.datetime.now()

print(x)
print(y)
print(f"Duration: {y - x}")