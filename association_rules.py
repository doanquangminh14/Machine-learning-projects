from apyori import apriori
import pandas as pd

df = pd.read_csv("dataset/assio.csv")
association_rules = apriori(df.values, min_support=0.50,min_lift=1.01) 

results = list(association_rules)

Items = []
Support = []
Items_base = []
Items_add = []
Confidence = []
Lift = []
for record in results:
    ordered_statistics = record[2]
    for stat in record[2]:
        Items.append(record[0])
        Support.append(record[1])
        Items_base.append(stat[0])
        Items_add.append(stat[1])
        Confidence.append(stat[2])
        Lift.append(stat[3])
        
df = pd.DataFrame({'Items':Items, 'Support':Support, 'Items_base':Items_base,
                   'Items_add':Items_add, 'Confidence':Confidence, 'Lift':Lift})

print(df)
