import argparse
import pprint


def parse_options(return_parser=False):
    # New CLI parser
    parser = argparse.ArgumentParser(description='Train deep implicit 3D geometry representations.')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--tooth-path', type=str,
                              help='Experiment name.')
    global_group.add_argument('--output-path', type=str, required=True)
    global_group.add_argument('--noise-threshold', type=int, default=2000,
                              help='threshold to delete the noise of mesh.')
    global_group.add_argument('--curvature-level', type=int, default=2,
                              help='the level of neighbors that will be used to get the average normal angel.')
    global_group.add_argument('--curvature-seg-threshold', type=float, default=0.2,
                              help='the percentage of points that will be used to remove surfaces.')
    global_group.add_argument('--refine-threshold', type=int, default=18000,                        
                              help='if the connected component\'s point number is larger than the thresshold, it will be splitted')
    global_group.add_argument('--removed-threshold', type=float, default=0.2,
                              help='the percentage of points that will be used to remove surfaces during refinement.')
    global_group.add_argument('--deleted-output-path', type=str, default='_result/deleted.obj',
                              help='Experiment name.')
    global_group.add_argument('--merge_step', type=int, default=5,
                              help='The steps of merge.')
    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)


def argparse_to_str(parser):
    """Convert parser to string representation for Tensorboard logging.

    Args:
        parser (argparse.parser): CLI parser
    """

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str

