"""Methods for parsing yn00 results files."""
import re

def parse_ng86(lines, results):
    if False:
        return 10
    'Parse the Nei & Gojobori (1986) section of the results.\n\n    Nei_Gojobori results are organized in a lower\n    triangular matrix, with the sequence names labeling\n    the rows and statistics in the format:\n    w (dN dS) per column\n    Example row (2 columns):\n    0.0000 (0.0000 0.0207) 0.0000 (0.0000 0.0421)\n    '
    sequences = []
    for line in lines:
        matrix_row_res = re.match('^([^\\s]+?)(\\s+-?\\d+\\.\\d+.*$|\\s*$|-1.0000\\s*\\(.*$)', line)
        if matrix_row_res is not None:
            line_floats_res = re.findall('-*\\d+\\.\\d+', matrix_row_res.group(2))
            line_floats = [float(val) for val in line_floats_res]
            seq_name = matrix_row_res.group(1).strip()
            sequences.append(seq_name)
            results[seq_name] = {}
            for i in range(0, len(line_floats), 3):
                NG86 = {}
                NG86['omega'] = line_floats[i]
                NG86['dN'] = line_floats[i + 1]
                NG86['dS'] = line_floats[i + 2]
                results[seq_name][sequences[i // 3]] = {'NG86': NG86}
                results[sequences[i // 3]][seq_name] = {'NG86': NG86}
    return (results, sequences)

def parse_yn00(lines, results, sequences):
    if False:
        while True:
            i = 10
    'Parse the Yang & Nielsen (2000) part of the results.\n\n    Yang & Nielsen results are organized in a table with\n    each row comprising one pairwise species comparison.\n    Rows are labeled by sequence number rather than by\n    sequence name.\n    '
    for line in lines:
        line_floats_res = re.findall('-*\\d+\\.\\d+', line)
        line_floats = [float(val) for val in line_floats_res]
        row_res = re.match('\\s+(\\d+)\\s+(\\d+)', line)
        if row_res is not None:
            seq1 = int(row_res.group(1))
            seq2 = int(row_res.group(2))
            seq_name1 = sequences[seq1 - 1]
            seq_name2 = sequences[seq2 - 1]
            YN00 = {}
            YN00['S'] = line_floats[0]
            YN00['N'] = line_floats[1]
            YN00['t'] = line_floats[2]
            YN00['kappa'] = line_floats[3]
            YN00['omega'] = line_floats[4]
            YN00['dN'] = line_floats[5]
            YN00['dN SE'] = line_floats[6]
            YN00['dS'] = line_floats[7]
            YN00['dS SE'] = line_floats[8]
            results[seq_name1][seq_name2]['YN00'] = YN00
            results[seq_name2][seq_name1]['YN00'] = YN00
            seq_name1 = None
            seq_name2 = None
    return results

def parse_others(lines, results, sequences):
    if False:
        return 10
    'Parse the results from the other methods.\n\n    The remaining methods are grouped together. Statistics\n    for all three are listed for each of the pairwise\n    species comparisons, with each method\'s results on its\n    own line.\n    The stats in this section must be handled differently\n    due to the possible presence of NaN values, which won\'t\n    get caught by my typical "line_floats" method used above.\n    '
    seq_name1 = None
    seq_name2 = None
    for line in lines:
        comp_res = re.match('\\d+ \\((.+)\\) vs. \\d+ \\((.+)\\)', line)
        if comp_res is not None:
            seq_name1 = comp_res.group(1)
            seq_name2 = comp_res.group(2)
        elif seq_name1 is not None and seq_name2 is not None:
            if 'dS =' in line:
                stats = {}
                line_stats = line.split(':')[1].strip()
                res_matches = re.findall('[dSNwrho]{1,3} =.{7,8}?', line_stats)
                for stat_pair in res_matches:
                    stat = stat_pair.split('=')[0].strip()
                    value = stat_pair.split('=')[1].strip()
                    try:
                        stats[stat] = float(value)
                    except ValueError:
                        stats[stat] = None
                if 'LWL85:' in line:
                    results[seq_name1][seq_name2]['LWL85'] = stats
                    results[seq_name2][seq_name1]['LWL85'] = stats
                elif 'LWL85m' in line:
                    results[seq_name1][seq_name2]['LWL85m'] = stats
                    results[seq_name2][seq_name1]['LWL85m'] = stats
                elif 'LPB93' in line:
                    results[seq_name1][seq_name2]['LPB93'] = stats
                    results[seq_name2][seq_name1]['LPB93'] = stats
    return results