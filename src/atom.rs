use std::collections::BTreeMap;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Atom(usize);

#[derive(Clone, Default)]
pub struct Table {
    str_to_id: BTreeMap<String, Atom>,
    id_to_str: Vec<String>,
}

impl Table {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, name: &str) -> Atom {
        if let Some(&atom) = self.str_to_id.get(name) {
            return atom;
        }

        let name = name.to_owned();

        let atom = Atom(self.id_to_str.len());
        self.id_to_str.push(name.clone());
        self.str_to_id.insert(name, atom);

        atom
    }

    pub fn resolve(&self, atom: Atom) -> String {
        self.id_to_str.get(atom.0)
            .unwrap_or_else(|| panic!("No such atom: {:?}", atom))
            .to_owned()
    }
}

