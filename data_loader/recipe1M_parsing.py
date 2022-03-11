import csv
import json


def create_recipe_file():
    file_name = "../data/recipe1M_layers/layer1.json"
    recipe_titles_rows = []
    output_csv_file = "../data/recipe_names_lower.csv"
    max_recipe_count = 1_000_000

    with open(file_name, 'r') as f:
        data = json.load(f)
        count = 0
        for item in data:
            recipe_titles_rows.append([item["title"].lower()])
            count += 1
            if count % 100_000 == 0:
                print(f"processed recipe number {count}")
            if count >= max_recipe_count:
                break

    with open(output_csv_file, mode='w') as writefile:
        csv_writer = csv.writer(writefile)
        csv_writer.writerow(['title'])
        csv_writer.writerows(recipe_titles_rows)


if __name__ == '__main__':
    create_recipe_file()
    print("done")
