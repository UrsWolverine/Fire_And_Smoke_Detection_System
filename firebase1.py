import firebase_admin
from firebase_admin import credentials, firestore, storage
# Init firebase with your credentials
cred = credentials.Certificate("fire-detection-bae7c-309d1a472cf3.json")
'''initialize_app(cred, {'storageBucket': '/'})
'''

# Put your local file path 
fileName = "Latest.jpg"
bucket = storage.bucket()
blob = bucket.blob(fileName)
blob.upload_from_filename(fileName)

# Opt : if you want to make public access from the URL
blob.make_public()

print("your file url", blob.public_url)