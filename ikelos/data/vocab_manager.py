from .vocabulary import Vocabulary
import os
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import warnings


class VocabManager(object):
    def __init__(self, mode=None):
        self.mode = mode
        self.mode_maps = {'nlp':self.prep}

    def add(self, key, new_content=None, filename=None, file_type="json"):
        if key in self.__dict__:
            if new_content:
                return list(self.get(key).imap(new_content))
            elif filename:
                exmsg2 = "Key exists ({}). Will not overwrite.".format(key)
                raise Exception("This situation is confusing. "+exmsg2)
            else:
                return []
        ### not in the dict.. technically an else condition
        ### but we return or raise in the last block.
        if new_content:
            new_v = Vocabulary.from_iterable(new_content, file_type=file_type)
        elif filename and "vocab" in filename.split("/")[-1]:
            new_v = Vocabulary.load(filename, file_type=file_type)
        elif filename:
            new_v = persistent_load(filename, True)
        else:
            new_v = Vocabulary(file_type=file_type)
        new_v.name = key
        self.__dict__[key] = new_v
        if self.mode:
            if self.mode not in self.mode_maps:
                warnings.warn("Vocab Manager has been set to an incorrect mode")
            else:
                self.mode_maps[self.mode](new_v)


    def prep(self, vocab):
        vocab.use_mask = True
        vocab.add(vocab.mask_symbol)
        vocab.add(vocab.unk_symbol)

    def get(self, key):
        """get item with key.

        purposely not emulating container type.
        potential headaches and obscurity later on.
        "simple is better can complex"
        "explicit is better than implicit"
        """
        return self.__dict__[key]

    def set(self, key, value):
        """set if we want to save other things
        """
        self.__dict__[key] = value

    def save(self, save_dir="", save_name="vocman.pkl"):
        config = {'members': []}
        for k,v in self.__dict__.items():
            if isinstance(v, Vocabulary):
                save_path = os.path.join(save_dir, "{}.vocab".format(k))
                v.save(save_path)
                print("saved {} at {}".format(k, save_path))
                config['members'].append((v.name,save_path,v.file_type))
            elif isinstance(v, np.ndarray):
                save_path = os.path.join(save_dir, "{}.mat".format(k))
                v.dump(save_path)
                print("saved {} at {}".format(k, save_path))
                warnings.warn("Currently saving numpy arrays; will not load them")
            elif isinstance(v, dict):
                save_path = os.path.join(save_dir, "{}_dict.pkl".format(k))
                try:
                    with open(save_path, "w") as fp:
                        pickle.dump(v, fp)
                    print("saved {} at {}".format(k, save_path))
                    warnings.warn("Currently saving dictionaries; will not load them")
                except Exception as e:
                    warnings.warn("dictionary saving failed: {} \n {}".format(k, e))
            else:
                warnings.warn("did not save {}".format(k))
                continue
            config[k] = save_path
        with open(os.path.join(save_dir, save_name), "w") as fp:
            pickle.dump(config, fp)

    def freeze_all(self, emit_unks=True):
        for k,v in self:
            if isinstance(v, Vocabulary):
                v.freeze(emit_unks)

    def __iter__(self):
        for k,v in self.__dict__.items():
            yield (k, v)

    @classmethod
    def load(cls, config_dict):
        vm = cls()
        members = config_dict['members']
        for name, filename, file_type in members:
            vm.add(name, filename=filename, file_type=file_type)
        return vm

    @classmethod
    def from_file(cls, filename, updated_filepath=None):
        with open(filename, 'rb') as fp:
            config = pickle.load(fp)
        my_dir = "/".join(filename.split("/")[:-1])
        update = lambda s: os.path.join(my_dir, s.split("/")[-1])
        config['members'] = [(name, update(filename), file_type) 
                             for name, filename, file_type in config['members']]
        return VocabManager.load(config)



def persistent_load(filename, raise_on_failure=False):
    with open(filename) as fp:
        try:
            return pickle.load(fp)
        except (KeyError, EOFError):
            pass
        try:
            return json.load(fp)
        except ValueError:
            pass
        if raise_on_failure:
            raise Exception("The file {} was not able to be opened with json or pickle".format(filename))
        else:
            return fp.readlines()




