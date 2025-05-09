#!/usr/bin/env python
# coding: utf-8

# In[21]:
import gspread
import pandas as pd


stop_words = [
    "co", "inc", "ltd", "group", "company", "com", "&", "corp", "corporation", 
    "plc", "pty", "llc", "gmbh", "ag", "sa", "bv", "sarl", 
    "associates", "partners", "international", "enterprise", "enterprises", "services", 
    "service", "consulting", "consultants", "industries",
    "holdings", "holding", "products", "productions", "systems", "designs", "design", 
    "management", "manufacturing", "global", "development", "developers", "investments", 
    "investment", "studio", "studios", "digital", "network", "networks", 
    "foundation", "trust", "associates", "union", "bank", "fund", "club", "society", 
    "cooperative", "coop", "limited", "unlimited", "brothers", "bros", "sons", "and", "construction", "tpo"
]

def get_worksheet_as_dataframe(spreadsheet, worksheet_name):
    worksheet = spreadsheet.worksheet(worksheet_name)
    records = worksheet.get_all_records()  # This returns a list of dictionaries
    return pd.DataFrame(records)

def update_google_sheet_with_dataframe(worksheet, dataframe):
    # Clearing the existing data in the worksheet
    worksheet.clear()

    # I convert the DataFrame to a list of lists, replacing NaN with an empty string
    rows = [dataframe.columns.tolist()] + dataframe.fillna('').values.tolist()

    # Then we update the worksheet in a single batch
    cell_range = 'A1:{}{}'.format(chr(64 + dataframe.shape[1]), dataframe.shape[0] + 1)
    worksheet.update(cell_range, rows)








# In[22]:


def clean_customer_name(name):
    # Trim the original name's leading and trailing whitespaces
    trimmed_original_name = name.strip()
    
    # Convert everything to lowercase
    name = trimmed_original_name.lower()
    
    # Remove punctuation and special characters 
    cleaned_name = re.sub(r'[^a-zA-Z\s]', '', name)
    
    # Split into tokens
    tokens = cleaned_name.split()
    
    # Remove stop words
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin and strip again
    cleaned_name = ' '.join(cleaned_tokens).strip()
    
    # Return the cleaned name if it's not empty; otherwise, return the trimmed original name
    return cleaned_name if cleaned_name else trimmed_original_name


# In[23]:


def aggressive_clean(name):
    # Optional. Made this just in case.
    # Convert to lowercase
    name = name.lower()
    
    # Remove punctuation and special characters 
    cleaned_name = re.sub(r'[^a-zA-Z\s]', '', name)
    
    # Split into tokens
    tokens = cleaned_name.split()
    
    # Remove stop words
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin and strip again
    cleaned_name = ' '.join(cleaned_tokens).strip()
    
    return cleaned_name


# In[24]:


def get_max_similarity_score(index, cosine_sim, threshold=0.8):
    # Cosine similarity scores to match customer names
    similar_indices = [i for i, sim in enumerate(cosine_sim[index]) if sim > threshold]
    
    # Check if similar_indices is not empty
    if similar_indices:
        return cosine_sim[index][similar_indices[0]]
    else:
        return 1.0  # Return 1.0 for rows that match with themselves


# In[25]:


def get_standard_name(index, cosine_sim, dataframe, column_name, threshold=0.8):
    similar_indices = [i for i, sim in enumerate(cosine_sim[index]) if sim > threshold]
    if similar_indices:
        return dataframe[column_name].iloc[similar_indices[0]].strip()
    else:
        return dataframe[column_name].iloc[index].strip()


# In[26]:


def get_similarity_score(index):
    similar_indices = [i for i, sim in enumerate(cosine_sim[index])]
    if similar_indices:
        return max([cosine_sim[index][i] for i in similar_indices])
    else:
        return 0


# In[34]:


import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ... stop words and helper functions ...

def process_customers(dataframe, column_name, manual_changes_file=None, 
                      manual_changes_worksheet_name=None, spreadsheet=None, 
                      output_worksheet_name=None):
    # Copy the dataframe
    df = dataframe.copy()
    
    # Make sure the columns actually exist.
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' is not a column in the provided DataFrame")
        
    df = df.sort_values(by=column_name, key=lambda x: x.str.len(), ascending=False)
    df = df.reset_index(drop=True)
    df['UNCLEAN_CUSTOMER'] = df[column_name]
    df['CLEAN_CUSTOMER'] = df['UNCLEAN_CUSTOMER'].apply(clean_customer_name)
    
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(df['CLEAN_CUSTOMER'])
    
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    df['MATCHED_NAME'] = df.index.map(lambda idx: get_standard_name(idx, cosine_sim, df, column_name))
    df['SIMILARITY_SCORE'] = df.index.map(lambda idx: get_max_similarity_score(idx, cosine_sim))
    
    if manual_changes_worksheet_name and spreadsheet:
        manual_changes_df = get_worksheet_as_dataframe(spreadsheet, manual_changes_worksheet_name)
        if 'Index' in manual_changes_df.columns:
            manual_changes_df.drop('Index', axis=1, inplace=True)
    elif manual_changes_file:
        manual_changes_df = pd.read_csv(manual_changes_file)
    else:
        manual_changes_df = None

    if manual_changes_df is not None:
        print("Columns in df:", df.columns)
        print("Columns in manual_changes_df:", manual_changes_df.columns)
        print('Columns in df:', df.columns)


        manually_changed_rows = df.merge(manual_changes_df, on='UNCLEAN_CUSTOMER', how='left', indicator=True)

        # Update MATCHED_NAME and SIMILARITY_SCORE based on manual changes
        mask = manually_changed_rows['_merge'] == 'both'
        df.loc[mask, 'MATCHED_NAME'] = manually_changed_rows.loc[mask, 'MATCHED_NAME_y']
        df.loc[mask, 'SIMILARITY_SCORE'] = manually_changed_rows.loc[mask, 'SIMILARITY_SCORE_y']

        # Trim to necessary columns
        df = df[['Customer', 'CLEAN_CUSTOMER', 'MATCHED_NAME', 'SIMILARITY_SCORE']]

        # Export clean names to a new sheet
        sheet_names = [sheet.title for sheet in spreadsheet.worksheets()]
        if output_worksheet_name in sheet_names:
            output_worksheet = spreadsheet.worksheet(output_worksheet_name)
            update_google_sheet_with_dataframe(output_worksheet, df)
        else:
            output_worksheet = spreadsheet.add_worksheet(title=output_worksheet_name, rows="1000", cols="20")
            set_with_dataframe(output_worksheet, df)

        return df

    customer_list = df.drop_duplicates(subset=column_name, keep='first')[[column_name, 'CLEAN_CUSTOMER', 'MATCHED_NAME', 'SIMILARITY_SCORE']].sort_values(by='CLEAN_CUSTOMER')
    # If you're working with a Google Sheets spreadsheet
    if output_worksheet_name and spreadsheet:
        output_worksheet = spreadsheet.worksheet(output_worksheet_name)
        update_google_sheet_with_dataframe(output_worksheet, customer_list)
    else:
        customer_list.to_csv('customer_list_v2.csv')
    
    return customer_list










