from collections import defaultdict, OrderedDict
import json
import os

import pandas as pd
import numpy as np

# print(os.getcwd())
RAW_CSV = 'services/foods_data.csv'
FILEPATH_EMBEDDINGS = 'services/ingredient_embeddings.csv'

# Function to calculate cosine similarity
def cosine_similarity(row1, row2):
    dot_product = np.dot(row1, row2)
    norm_row1 = np.linalg.norm(row1)
    norm_row2 = np.linalg.norm(row2)
    return dot_product / (norm_row1 * norm_row2)

# Function to find most similar rows
def find_similar_rows(df, ingredients, num_similar=10):
    target_row = df[df.index.str.contains('|'.join(ingredients), na=False)].mean()
    # target_row = df[df["Unnamed: 0"].str.contains('|'.join(ingredients), na=False)].mean()

    # target_row = dataframe[any(dataframe.index in ingredients)].mean()
    # target_row = dataframe.iloc[target_row_index]

    # Calculate cosine similarity of the target row with all other rows
    similarities = [
        cosine_similarity(target_row[1:], row[1:]) 
        for _, row in df.iterrows()
    ]

    # Get indices of top similar rows (excluding the target row itself)
    top_similar_indices = np.argsort(similarities)[::-1][1:num_similar + 1]
    top_similar_indices = [(i, similarities[i]) for i in top_similar_indices]

    # Return the top similar rows
    return top_similar_indices


class RecipeRecommedModel:
    def __init__(self):
        self.model = None
        self.embeddings = None
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

        if not os.path.exists(FILEPATH_EMBEDDINGS):
            from gensim.models import Word2Vec
            model = Word2Vec([ingredients for ingredients in self.data['main_ing']], vector_size=100, window=5,  min_count=1, workers=4)
            # get node name and embedding vector index.
            # init dataframe using embedding vectors and set index as node name
            name_index = np.array([(v, i) for i, v in enumerate(model.wv.index_to_key)])
            df =  pd.DataFrame(model.wv.vectors[name_index[:,1].astype(int)])
            df.index = name_index[:, 0]
            df.to_csv(FILEPATH_EMBEDDINGS)
            self.embeddings = df

        else:
            self.embeddings = pd.read_csv(FILEPATH_EMBEDDINGS).set_index("Unnamed: 0")
        
        print('Success to set model')

    def recommend_recipes(self, food: str=None, ingredients: list[str]=None):
        # if self.model is None:
        #     print("Model is None")
        #     return

        if self.data is None:
            print("Data is None")
            return
        
        if food:
            if food not in self.data['name'].values:
                print("Food is not in data")
                return
            
            selected_food = self.data[self.data['name'] == food]
            selected_ingredients = selected_food['main_ing'].values[0]
        elif ingredients:
            selected_ingredients = ingredients
        
        
        similar_rows = find_similar_rows(self.embeddings, selected_ingredients, num_similar=5)
        similar_ingredients = [(self.embeddings.index[i],p) for i,p in similar_rows]
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
                
                if ingredient in selected_ingredients:
                    recipe_scores[temp_food['name']] += 1

        ranked_recipes = sorted(recipe_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_recipes

    def convert_to_json(self, recommend_results, recommend_len) -> dict:
        recommend_len = min(recommend_len, len(recommend_results))

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

            # temp_json_list.append(json.dumps(file_data, ensure_ascii=False, indent="\t"))
            temp_json_list.append(file_data)

        return temp_json_list

rcmd_model = RecipeRecommedModel()
rcmd_model.set_init()

def main(event, context):
    params = event.get('queryStringParameters')

    # if params:
    #     search = params.get('search')
    # else:
    #     search = None

    # result_list = rcmd_model.recommend_recipes(search)
    
    if params.get('menu'):
        menu = params.get('menu')
        result_list = rcmd_model.recommend_recipes(food=menu)
        json_list = rcmd_model.convert_to_json(result_list, 10)
    
    elif params.get('ingredients'):
        ingredients = params.get('ingredients').split(",")
        result_list = rcmd_model.recommend_recipes(ingredients=ingredients)
        json_list = rcmd_model.convert_to_json(result_list, 10)

    else:
        json_list = []

    response = {
        "statusCode": 200,
        "body": json.dumps(json_list),
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

    # while True:
    # result_list = rcmd_model.recommend_recipes(food='짬뽕')
    result_list = rcmd_model.recommend_recipes(ingredients=['팔각','파스타면'])
    json_list = rcmd_model.convert_to_json(result_list, 10)
    print(result_list)
    print(json_list[0])

