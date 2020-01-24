
def field_name_changer(df, columns_name_list):
    for value in columns_name_list[:-1]:
        df[value] = df[value].apply(lambda x: str(x) + value)
