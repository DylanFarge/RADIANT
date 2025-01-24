from flask_login import UserMixin
import json, os, shutil
from assets.status_codes import Database as c
import pandas as pd
from processing import data

# User data model. It has to have at least self.id as a minimum
class User(UserMixin):
    def __init__(self, username):
        '''UserMixin allows for the additional methods:
        - is_authenticated
        - is_active
        - is_anonymous
        - get_id'''
        self.id = username
        self.df = pd.read_json(f"src/database/{username}/df.json")
        self.an = pd.read_json(f"src/database/{username}/analysis.json")
        self.colours = data.analysis_colours(set=True, df=self.df)
        
class Credentials:
    with open("src/database/credentials.json", "r") as f:
        creds = json.load(f)

    def user_not_found(username, password):
        '''
        0 - success
        1 - username not found
        2 - password incorrect
        '''
        if username not in Credentials.creds:
            return c.USERNAME_NOT_FOUND
        if password not in Credentials.creds[username]:
            return c.PASSWORD_INCORRECT
        return c.SUCCESS
    
    def create_user(username, password):
        # add user to credentials.json
        if username in Credentials.creds:
            return False
        Credentials.creds[username] = password
        with open("src/database/credentials.json", "w") as f:
            json.dump(Credentials.creds, f)

        os.mkdir(f"src/database/{username}")
        shutil.copy("src/database/official/analysis.json", f"src/database/{username}/analysis.json")
        shutil.copy("src/database/official/df.json", f"src/database/{username}/df.json")

        print("User created")
        return True        
    
    def delete_account(username):
        print(f"DELETING ACCOUNT {username}...",end="")
        Credentials.creds.pop(username)
        with open("src/database/credentials.json", "w") as f:
            json.dump(Credentials.creds, f)
        shutil.rmtree(f"src/database/{username}")
        print("DONE")