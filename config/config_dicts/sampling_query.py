sampling_config = {"interrupt_name": ["sampling"],
"interrupt_message":  """
    > ⚠️ Your dataset has {n_rows} rows. The recommended 20% sample would be {n_rows_20} rows, which may be slow or memory-intensive.\n\n
    **Would you like to use a smaller 10% sample ({n_rows_10} rows) instead?**\n\n
    """

}
