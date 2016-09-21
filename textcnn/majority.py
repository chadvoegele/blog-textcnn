def majority_elements(stream, n):
    counts = {}
    for element in stream:
        __update_counts(counts, element, n)

    return counts

def __update_counts(counts, element, n):
    if element in counts:
        counts[element] += 1
        return

    if len(counts) < n:
        counts[element] = 1
        return

    for e in counts:
        counts[e] -= 1

    delete_keys = [e for (e,v) in counts.items() if v == 0]
    for key in delete_keys:
        del counts[key]

    if len(counts) < n:
        counts[element] = 1
