#%%
from notion_client import Client
from pprint import pprint


#%%
# integration in Notion

notion_token_230808 = 'secret_gkAXWxlfrzEhCvOVRoBOk0BpVP4nKhfp4nb1y4cW13y'
results_page_id = '92597b4e6b714ba9ad85adabd908e3a3'
results_database_id = '5d836d0ca4354f4485f4ea8d10b047f9'


def write_text(notion_page_id, text):
    client = Client(auth=notion_token_230808)   

    client.blocks.children.append(
        block_id = notion_page_id,
        children=[
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": text
                            }
                            ,
                            "annotations": {
                                
                            }
                        }
                    ]
                }
            }
        ]
    )

def read_text(client, notion_page_id):
    response = client.blocks.children.list(block_id=notion_page_id)

    return response['results']

def create_blocks_from_content(client, content):

    page_simple_blocks = []

    for block in content:
        block_id = block['id']
        block_type = block['type']
        block_has_children = block['has_children']
        richt_text = block[block_type]['rich_text']

        if not richt_text:
            continue

        simple_block = {
            'id': block_id,
            'type': block_type,
            'text': richt_text[0]['plain_text']
        }

        page_simple_blocks.append(simple_block)

        if block_has_children:
            nested_children = read_text(client, block_id)
            simple_block['children'] = create_blocks_from_content(client, nested_children)
        
        

    return page_simple_blocks

def safe_get(data, dot_chained_keys):
    '''
        {'a': {'b': [{'c': 1}]}}
        safe_get(data, 'a.b.0.c') -> 1
    '''
    keys = dot_chained_keys.split('.')
    for key in keys:
        try:
            if isinstance(data, list):
                data = data[int(key)]
            else:
                data = data[key]
        except (KeyError, TypeError, IndexError):
            return None
    return data

def write_row_result_database(client, database_id, row_result):

    client.pages.create(
        **{
            "parent": { "database_id": database_id },
            "properties": {
                "Version": {'title': [{'text': {'content': row_result['Version']}}]},
                'Parameter': {'rich_text': [{'text': {'content': row_result['Parameter']}}]},
                'Date': {'date': {'start': row_result['Date']}},
                'Results': {'rich_text': [{'text': {'content': row_result['Results']}}]},
                'Transducer': {'select': {'name': row_result['Transducer']}},
                'Cropping properties': {'rich_text': [{'text': {'content': row_result['Cropping properties']}}]},
                'Window function': {'rich_text': [{'text': {'content': row_result['Window function']}}]}
            }
        }
    )

def safe_to_notion_def(version, notion_evaluation_parameter, notion_results, notion_transducer,notion_cropping_properties, notion_window_func, safe_to_notion, database_id):
    if safe_to_notion == True:
        # define Notion IDs 
        notion_token_230808 = 'secret_gkAXWxlfrzEhCvOVRoBOk0BpVP4nKhfp4nb1y4cW13y'
        results_page_id = '92597b4e6b714ba9ad85adabd908e3a3'
        client = Client(auth=notion_token_230808)
    
        row_results = create_row_result_database(version, notion_evaluation_parameter, notion_results, notion_transducer, notion_cropping_properties, notion_window_func)

        write_row_result_database(client, database_id, row_results)

def create_row_result_database(version, parameter, results, transducer, cropping_properties, window_function):
    from datetime import date

    row_result={
        'Version': version,
        'Parameter': parameter,
        'Date': str(date.today()),
        'Results': results,
        'Transducer': transducer,
        'Cropping properties': cropping_properties,
        'Window function': window_function
    }

    return row_result


    
# %%