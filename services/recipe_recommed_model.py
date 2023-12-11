import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict, OrderedDict
import json

RAW_CSV = 'news-api-server/services/foods_data.csv'

class RecipeRecommedModel:
    def __init__(self):
        self.model = None
        self.data = None

    def set_init(self):
        
        self.load_data(RAW_CSV)
        self.set_model()

    def load_data(self, path):

        if path != None:
            self.data = pd.read_csv(path)
        else:
            self.data = pd.read_csv(RAW_CSV)

        if "main_ing" in self.data.columns:
            self.data['main_ing'] = self.data['main_ing'].apply(lambda x: x.split(','))
            self.data['main_ing'] = self.data['main_ing'].apply(lambda x: [i.replace("'",'') for i in x])
            self.data['main_ing'] = self.data['main_ing'].apply(lambda x: [i.replace("[",'') for i in x])
            self.data['main_ing'] = self.data['main_ing'].apply(lambda x: [i.replace("]",'') for i in x])

            self.data['total_ingredient'] = self.data['total_ingredient'].apply(lambda x: x[1:-1].split(','))
            self.data['total_ingredient'] = self.data['total_ingredient'].apply(lambda x: [i.replace("'",'') for i in x])
            self.data['total_ingredient'] = self.data['total_ingredient'].apply(lambda x: [i.replace("[",'') for i in x])
            self.data['total_ingredient'] = self.data['total_ingredient'].apply(lambda x: [i.replace("]",'') for i in x])

            self.data['steps'] = self.data['steps'].apply(lambda x: x[1:-1].split("',"))
            self.data['steps'] = self.data['steps'].apply(lambda x: [i.replace("'",'') for i in x])
            self.data['steps'] = self.data['steps'].apply(lambda x: [i.replace("[",'') for i in x])
            self.data['steps'] = self.data['steps'].apply(lambda x: [i.replace("]",'') for i in x])

            self.data['tags'] = self.data['tags'].apply(lambda x: x[1:-1].split(','))
            self.data['tags'] = self.data['tags'].apply(lambda x: [i.replace("'",'') for i in x])
            self.data['tags'] = self.data['tags'].apply(lambda x: [i.replace("[",'') for i in x])
            self.data['tags'] = self.data['tags'].apply(lambda x: [i.replace("]",'') for i in x])

        else:
            print("Raw Data Error")
            return

    def set_model(self):

        if self.data is None:
            print("Data is None")
            return
        
        self.model = Word2Vec([ingredients for ingredients in self.data['main_ing']], vector_size=100, window=5,  min_count=1, workers=4)
        print('Success to set model')

    def recommend_recipes(self, food):
        if self.model is None:
            print("Model is None")
            return

        if self.data is None:
            print("Data is None")
            return
        
        if food not in self.data['name'].values:
            print("Food is not in data")
            return
        
        selected_food = self.data[self.data['name'] == food]
        selected_ingredient = selected_food['main_ing'].values[0]
        
        similar_ingredients = self.model.wv.most_similar(positive=selected_ingredient)
        similar_ingredients = dict(similar_ingredients)

        recipe_scores = defaultdict(int)

        for i in range(len(self.data)):
            temp_food = self.data.iloc[i]
            main_ingredient = temp_food['main_ing']

            if temp_food['name'] == food:
                continue

            for ingredient in main_ingredient:
                if ingredient in similar_ingredients.keys():
                    recipe_scores[temp_food['name']] += similar_ingredients[ingredient]
                
                if ingredient in selected_ingredient:
                    recipe_scores[temp_food['name']] += 1.0

        ranked_recipes = sorted(recipe_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_recipes

    def convert_to_json(self, recommend_results, recommend_len):

        temp_json_list = []
        length = max(len(recommend_results), recommend_len)

        for i in range(length):
            result_food = self.data[self.data['name'] == recommend_results[i][0]]

            file_data = OrderedDict()
            file_data["id"] = str(result_food['name'].index[0])
            file_data["name"] = result_food['name'].values[0]
            file_data["description"] = result_food['description'].values[0]
            file_data["steps"] = result_food['steps'].values[0]
            file_data["tags"] = result_food['tags'].values[0]
            file_data["ingredients"] = result_food['total_ingredient'].values[0]

            temp_json_list.append(json.dumps(file_data, ensure_ascii=False, indent="\t"))

        return temp_json_list


rcmd_model = RecipeRecommedModel()
rcmd_model.set_init()

def main(event, context):
    params = event.get('queryStringParameters')

    if params:
        search = params.get('search')
    else:
        search = None

    result_list = rcmd_model.recommend_recipes(search)

    # body = {
    #     'trends': trends,
    # }

    # print('body')

    response = {
        "statusCode": 200,
        "body": json.dumps(result_list),
        # CORS
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": True,
        }
    }

    return response

if __name__ == '__main__':

    rcmd_model = RecipeRecommedModel()
    rcmd_model.set_init()

    user_selected_food = '돼지안심 스테이크'

    result_list = rcmd_model.recommend_recipes(user_selected_food)
    print(result_list)
    json_list = rcmd_model.convert_to_json(result_list, 10)
    print(json_list[0])

    

