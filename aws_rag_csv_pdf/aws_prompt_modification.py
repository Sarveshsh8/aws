import pandas as pd




# chekc this out to get a datarame

json_output = {
    "Answer": "To analyze the gender distribution across the data, we can generate a count of students by gender and visualize it in a bar chart.",
    "SQL Query": "SELECT gender, COUNT(*) as count FROM students GROUP BY gender",
    "SQL Query Answer": {
        "columns": ["gender", "count"],
        "data": [
            ["male", 8],
            ["female", 4]
        ]
    },
    "Visualization": {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Gender Distribution",
        "data": {
            "values": [
                {"gender": "male", "count": 8},
                {"gender": "female", "count": 4}
            ]
        },
        "mark": "bar",
        "encoding": {
            "x": {"field": "gender", "type": "nominal"},
            "y": {"field": "count", "type": "quantitative"},
            "color": {"field": "gender", "type": "nominal"}
        }
    }
}


ans = json_output.get('SQL Query Answer',None)


df = pd.DataFrame(ans['data'], columns=ans['columns'])

# df = pd.DataFrame(ans)


#   await cl.Message(content=df.to_markdown()).send()

#

# await cl.Table(data=df.values.tolist(), column_names=df.columns.tolist()).send()
########