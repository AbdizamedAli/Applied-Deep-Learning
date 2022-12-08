from pathlib import Path
import argparse
def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    models = ['Shallow','Deep','Deep_Batch_Normalisation']
    tb_log_dir_prefix = f'Epochs{args.epochs}_Model_Type{models[args.model]}_Batch_Size{args.batch_size}_run_'

    i = 0
    while i < 1000:
        tb_log_dir = Path("logs") / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)