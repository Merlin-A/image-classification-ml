    Result = __class_number_to_name[classify_image(get_b64_image(), None)[0]].split('_')
    Result = str(Result[0]).capitalize() + ' ' + str(Result[1]).capitalize()
    print("The Image is of -> ",Result)