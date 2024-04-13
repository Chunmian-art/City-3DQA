import os

main_path = "/workspace/Datasets/UrbanBIS/UrbanBIS/CityQA/"

l = []
for city_name in os.listdir(main_path):
    for area_name in os.listdir(f"{main_path}/{city_name}"):
        print(area_name)
        l.append(f"{city_name}_{area_name}".replace(".txt", ""))

with open("/workspace/UrbanQA/data/urbanbis/meta_data/city_area.txt", "w") as file:
    for item in l:
        file.write("%s\n" % item)