import os
import pickle

class Logger:
    """
    Simple logger used to save scalars with tensorboard style functions.
    """
    def __init__(self, log_path="./"):
        self.log_path = log_path
        self.tags = {}

    def _save(self):
        for tag in self.tags:
            # adopt tensorboard syntax for grouping tags
            filename = f"{self.log_path}{tag}.p"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pickle.dump(self.tags[tag], open(filename, 'wb'))

    def make_new_tag(self, tag):
        assert tag not in self.tags, AttributeError("Tag already exists")
        self.tags[tag] = {"scalars": [], "global_step": [],  "walltime": []}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """
        Add single entry to dicts
        """
        # get the tag dictionary from the object if it exists. Else return empty template dict
        if tag not in self.tags:
            self.make_new_tag(tag)

        self.tags[tag]['scalars'].append(scalar_value)
        self.tags[tag]['global_step'].append(global_step)
        self.tags[tag]['walltime'].append(walltime)

        self._save()

    def close(self):
        self._save()


if __name__ == "__main__":
    writer = Logger()

    x = range(100)
    for i in x:
        writer.add_scalar('test/losses', i * 2, i)
    writer.close()

