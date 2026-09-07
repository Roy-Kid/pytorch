global_flag = False
delete_global_value = True


def set_flag_true():
    global global_flag
    global_flag = True


def set_flag_false():
    global global_flag
    global_flag = False


def delete_global_value_fn():
    global delete_global_value
    del delete_global_value


def delete_missing_global_value_fn():
    global delete_missing_global_value
    del delete_missing_global_value
