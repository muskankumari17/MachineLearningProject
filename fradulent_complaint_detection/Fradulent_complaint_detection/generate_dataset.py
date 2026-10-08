import pandas as pd
import random

genuine_templates = [
    "The {area} is not working properly.",
    "There is an issue with {facility}.",
    "The {service} process is delayed.",
    "Students are facing problems due to {issue}.",
    "The {area} needs maintenance."
]

fake_templates = [
    "Marks are being changed intentionally in {area}.",
    "The administration is secretly manipulating {service}.",
    "Funds for {facility} are being misused.",
    "There is favoritism in {area}.",
    "The system is designed to fail students in {service}."
]

areas = ["Library", "Computer Lab", "Examination Cell", "Hostel Block A", "Accounts Section"]
services = ["scholarship", "exam registration", "attendance update", "result processing"]
facilities = ["WiFi system", "water supply", "electricity system", "biometric system"]
issues = ["slow internet", "power cuts", "incorrect attendance", "delay in results"]

data = []

for _ in range(250):
    complaint = random.choice(genuine_templates).format(
        area=random.choice(areas),
        service=random.choice(services),
        facility=random.choice(facilities),
        issue=random.choice(issues)
    )
    data.append([complaint, "Genuine"])

for _ in range(250):
    complaint = random.choice(fake_templates).format(
        area=random.choice(areas),
        service=random.choice(services),
        facility=random.choice(facilities)
    )
    data.append([complaint, "Fake"])

random.shuffle(data)

df = pd.DataFrame(data, columns=["complaint", "label"])
df.to_csv("complaints.csv", index=False)

print("Dataset created successfully!")