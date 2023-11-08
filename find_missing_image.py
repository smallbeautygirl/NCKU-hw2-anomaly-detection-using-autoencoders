import os

disappear = []

b = [
    f.split(".")[0].lstrip("0")
    for f in os.listdir(f"TA-Class8/Test/NonDefect")
    if f.split(".")[1] == "PNG"
]

for x in range(1, 1150):
    if str(x) not in b:
        disappear.append(x)

print(disappear)  # 會印出 [120]
