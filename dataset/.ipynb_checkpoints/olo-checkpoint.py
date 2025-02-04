def load_data():
    data = {
        'valence': [],
        'year': [],
        'acousticness': [],
        'artists': [],
        'danceability': [],
        'duration_ms': [],
        'energy': [],
        'explicit': [],
        'id': [],
        'instrumentalness': [],
        'key':[],
        'liveness':[],
        'loudness': [],
        'mode': [],
        'name':[],
        'popularity':[], 
        'release_date': [],
        'speechiness':[],
        'tempo':[]
    }

    field_order = ['valence', 'year', 'acousticness', 'artists', 'danceability', 'duration_ms','energy',
                  'explicit', 'id','instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'name',
                  'popularity', 'release_date', 'speechiness', 'tempo']

      # Read and process the file
    with open('./data.csv', 'r',  encoding='utf-8') as file:
        header = next(file)
        for line in file:
            # Split the line by commas, handling quotes for the artists and name fields
            parts = []
            buffer = ""
            inside_quotes = False
            for char in line.strip():
                if char == '"':
                    inside_quotes = not inside_quotes
                elif char == ',' and not inside_quotes:
                    parts.append(buffer)
                    buffer = ""
                else:
                    buffer += char
            parts.append(buffer)  # Append the last part
            # print(parts)
                # Map the parts to their respective fields
            for i, field in enumerate(field_order):
                value = parts[i]
        
                    # Convert types where necessary
                if field in ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo']:
                    value = float(value)
                elif field in ['year', 'duration_ms', 'key', 'popularity', 'explicit', 'mode']:
                    value = int(value)
                elif field == 'artists' or field == 'name':
                    value = value.strip('"').strip("[]").split(', ')
                    value = [v.strip("'") for v in value]
                elif field == 'loudness':
                    value = float(value)
                elif field == 'release_date':
                    value = value.strip('"')
        
                # Append the value to the appropriate list in the dictionary
                data[field].append(value)
        
                # Print the result to verify
    
    return data
           


tamuna_jaounda = load_data()


print(tamuna_jaounda['valence'][1173])