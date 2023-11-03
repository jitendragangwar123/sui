// Copyright (c) The Diem Core Contributors
// Copyright (c) The Move Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::{
    expansion::ast::{Address, ModuleIdent, ModuleIdent_, SpecId},
    hlir::ast::{self as H, Var},
    parser::ast::{ConstantName, DatatypeName, FunctionName, VariantName},
    shared::{CompilationEnv, NumericalAddress},
};
use move_core_types::account_address::AccountAddress as MoveAddress;
use move_ir_types::ast as IR;
use move_symbol_pool::Symbol;
use std::{
    clone::Clone,
    collections::{BTreeMap, BTreeSet, HashMap},
};
use IR::Ability;

/// Compilation context for a single compilation unit (module).
/// Contains all of the dependencies actually used in the module
pub struct Context<'a> {
    pub env: &'a mut CompilationEnv,
    current_package: Option<Symbol>,
    current_module: Option<&'a ModuleIdent>,
    seen_structs: BTreeSet<(ModuleIdent, DatatypeName)>,
    seen_functions: BTreeSet<(ModuleIdent, FunctionName)>,
    spec_info: BTreeMap<SpecId, (IR::NopLabel, BTreeMap<Var, H::SingleType>)>,
}

impl<'a> Context<'a> {
    pub fn new(
        env: &'a mut CompilationEnv,
        current_package: Option<Symbol>,
        current_module: Option<&'a ModuleIdent>,
    ) -> Self {
        Self {
            env,
            current_package,
            current_module,
            seen_structs: BTreeSet::new(),
            seen_functions: BTreeSet::new(),
            spec_info: BTreeMap::new(),
        }
    }

    #[allow(unused)]
    pub fn current_package(&self) -> Option<Symbol> {
        self.current_package
    }

    pub fn current_module(&self) -> Option<&'a ModuleIdent> {
        self.current_module
    }

    fn is_current_module(&self, m: &ModuleIdent) -> bool {
        self.current_module.map(|cur| cur == m).unwrap_or(false)
    }

    pub fn finish_function(
        &mut self,
    ) -> BTreeMap<SpecId, (IR::NopLabel, BTreeMap<Var, H::SingleType>)> {
        std::mem::take(&mut self.spec_info)
    }

    //**********************************************************************************************
    // Dependency item building
    //**********************************************************************************************

    pub fn materialize(
        self,
        dependency_orderings: &HashMap<ModuleIdent, usize>,
        struct_declarations: &HashMap<
            (ModuleIdent, DatatypeName),
            (BTreeSet<IR::Ability>, Vec<IR::DatatypeTypeParameter>),
        >,
        function_declarations: &HashMap<
            (ModuleIdent, FunctionName),
            (BTreeSet<(ModuleIdent, DatatypeName)>, IR::FunctionSignature),
        >,
    ) -> (Vec<IR::ImportDefinition>, Vec<IR::ModuleDependency>) {
        let Context {
            current_module: _current_module,
            mut seen_structs,
            seen_functions,
            ..
        } = self;
        let mut module_dependencies = BTreeMap::new();
        Self::function_dependencies(
            function_declarations,
            &mut module_dependencies,
            &mut seen_structs,
            seen_functions,
        );
        Self::struct_dependencies(struct_declarations, &mut module_dependencies, seen_structs);
        let mut imports = vec![];
        let mut ordered_dependencies = vec![];
        for (module, (structs, functions)) in module_dependencies {
            let dependency_order = dependency_orderings[&module];
            let ir_name = Self::ir_module_alias(&module);
            let ir_ident = Self::translate_module_ident(module);
            imports.push(IR::ImportDefinition::new(ir_ident, Some(ir_name)));
            ordered_dependencies.push((
                dependency_order,
                IR::ModuleDependency {
                    name: ir_name,
                    datatypes: structs,
                    functions,
                },
            ));
        }
        ordered_dependencies.sort_by_key(|(ordering, _)| *ordering);
        let dependencies = ordered_dependencies.into_iter().map(|(_, m)| m).collect();
        (imports, dependencies)
    }

    fn insert_struct_dependency(
        module_dependencies: &mut BTreeMap<
            ModuleIdent,
            (Vec<IR::DatatypeDependency>, Vec<IR::FunctionDependency>),
        >,
        module: ModuleIdent,
        struct_dep: IR::DatatypeDependency,
    ) {
        module_dependencies
            .entry(module)
            .or_insert_with(|| (vec![], vec![]))
            .0
            .push(struct_dep);
    }

    fn insert_function_dependency(
        module_dependencies: &mut BTreeMap<
            ModuleIdent,
            (Vec<IR::DatatypeDependency>, Vec<IR::FunctionDependency>),
        >,
        module: ModuleIdent,
        function_dep: IR::FunctionDependency,
    ) {
        module_dependencies
            .entry(module)
            .or_insert_with(|| (vec![], vec![]))
            .1
            .push(function_dep);
    }

    fn struct_dependencies(
        struct_declarations: &HashMap<
            (ModuleIdent, DatatypeName),
            (BTreeSet<Ability>, Vec<IR::DatatypeTypeParameter>),
        >,
        module_dependencies: &mut BTreeMap<
            ModuleIdent,
            (Vec<IR::DatatypeDependency>, Vec<IR::FunctionDependency>),
        >,
        seen_structs: BTreeSet<(ModuleIdent, DatatypeName)>,
    ) {
        for (module, sname) in seen_structs {
            let struct_dep = Self::struct_dependency(struct_declarations, &module, sname);
            Self::insert_struct_dependency(module_dependencies, module, struct_dep);
        }
    }

    fn struct_dependency(
        struct_declarations: &HashMap<
            (ModuleIdent, DatatypeName),
            (BTreeSet<Ability>, Vec<IR::DatatypeTypeParameter>),
        >,
        module: &ModuleIdent,
        sname: DatatypeName,
    ) -> IR::DatatypeDependency {
        let key = (*module, sname);
        let (abilities, type_formals) = struct_declarations.get(&key).unwrap().clone();
        let name = Self::translate_struct_name(sname);
        IR::DatatypeDependency {
            abilities,
            name,
            type_formals,
        }
    }

    fn function_dependencies(
        function_declarations: &HashMap<
            (ModuleIdent, FunctionName),
            (BTreeSet<(ModuleIdent, DatatypeName)>, IR::FunctionSignature),
        >,
        module_dependencies: &mut BTreeMap<
            ModuleIdent,
            (Vec<IR::DatatypeDependency>, Vec<IR::FunctionDependency>),
        >,
        seen_structs: &mut BTreeSet<(ModuleIdent, DatatypeName)>,
        seen_functions: BTreeSet<(ModuleIdent, FunctionName)>,
    ) {
        for (module, fname) in seen_functions {
            let (functions_seen_structs, function_dep) =
                Self::function_dependency(function_declarations, &module, fname);
            Self::insert_function_dependency(module_dependencies, module, function_dep);
            seen_structs.extend(functions_seen_structs)
        }
    }

    fn function_dependency(
        function_declarations: &HashMap<
            (ModuleIdent, FunctionName),
            (BTreeSet<(ModuleIdent, DatatypeName)>, IR::FunctionSignature),
        >,
        module: &ModuleIdent,
        fname: FunctionName,
    ) -> (
        BTreeSet<(ModuleIdent, DatatypeName)>,
        IR::FunctionDependency,
    ) {
        let key = (*module, fname);
        let (seen_structs, signature) = function_declarations.get(&key).unwrap().clone();
        let name = Self::translate_function_name(fname);
        (seen_structs, IR::FunctionDependency { name, signature })
    }

    //**********************************************************************************************
    // Name translation
    //**********************************************************************************************

    fn ir_module_alias(sp!(_, ModuleIdent_ { address, module }): &ModuleIdent) -> IR::ModuleName {
        let s = match address {
            Address::Numerical {
                value: sp!(_, a_), ..
            } => format!("{:X}::{}", a_, module),
            Address::NamedUnassigned(name) => format!("{}::{}", name, module),
        };
        IR::ModuleName(s.into())
    }

    pub fn resolve_address(&self, addr: Address) -> NumericalAddress {
        addr.into_addr_bytes()
    }

    pub fn translate_module_ident(
        sp!(_, ModuleIdent_ { address, module }): ModuleIdent,
    ) -> IR::ModuleIdent {
        let address_bytes = address.into_addr_bytes();
        let name = Self::translate_module_name_(module.0.value);
        IR::ModuleIdent::new(name, MoveAddress::new(address_bytes.into_bytes()))
    }

    fn translate_module_name_(s: Symbol) -> IR::ModuleName {
        IR::ModuleName(s)
    }

    fn translate_struct_name(n: DatatypeName) -> IR::DatatypeName {
        IR::DatatypeName(n.0.value)
    }

    fn translate_enum_name(n: DatatypeName) -> IR::DatatypeName {
        IR::DatatypeName(n.0.value)
    }

    fn translate_variant_name(n: VariantName) -> IR::VariantName {
        IR::VariantName(n.0.value)
    }

    fn translate_constant_name(n: ConstantName) -> IR::ConstantName {
        IR::ConstantName(n.0.value)
    }

    fn translate_function_name(n: FunctionName) -> IR::FunctionName {
        IR::FunctionName(n.0.value)
    }

    //**********************************************************************************************
    // Name resolution
    //**********************************************************************************************

    pub fn struct_definition_name(&self, m: &ModuleIdent, s: DatatypeName) -> IR::DatatypeName {
        assert!(
            self.is_current_module(m),
            "ICE invalid struct definition lookup"
        );
        Self::translate_struct_name(s)
    }

    pub fn enum_definition_name(&self, m: &ModuleIdent, e: DatatypeName) -> IR::DatatypeName {
        assert!(
            self.is_current_module(m),
            "ICE invalid enum definition lookup"
        );
        Self::translate_struct_name(e)
    }

    pub fn variant_name(&self, v: VariantName) -> IR::VariantName {
        Self::translate_variant_name(v)
    }

    pub fn qualified_struct_name(
        &mut self,
        m: &ModuleIdent,
        s: DatatypeName,
    ) -> IR::QualifiedDatatypeIdent {
        let mname = if self.is_current_module(m) {
            IR::ModuleName::module_self()
        } else {
            self.seen_structs.insert((*m, s));
            Self::ir_module_alias(m)
        };
        let n = Self::translate_struct_name(s);
        IR::QualifiedDatatypeIdent::new(mname, n)
    }

    pub fn function_definition_name(&self, m: &ModuleIdent, f: FunctionName) -> IR::FunctionName {
        assert!(
            self.current_module == Some(m),
            "ICE invalid function definition lookup"
        );
        Self::translate_function_name(f)
    }

    pub fn qualified_function_name(
        &mut self,
        m: &ModuleIdent,
        f: FunctionName,
    ) -> (IR::ModuleName, IR::FunctionName) {
        let mname = if self.is_current_module(m) {
            IR::ModuleName::module_self()
        } else {
            self.seen_functions.insert((*m, f));
            Self::ir_module_alias(m)
        };
        let n = Self::translate_function_name(f);
        (mname, n)
    }

    pub fn constant_definition_name(&self, m: &ModuleIdent, f: ConstantName) -> IR::ConstantName {
        assert!(
            self.current_module == Some(m),
            "ICE invalid constant definition lookup"
        );
        Self::translate_constant_name(f)
    }

    pub fn constant_name(&mut self, f: ConstantName) -> IR::ConstantName {
        Self::translate_constant_name(f)
    }

    //**********************************************************************************************
    // Nops
    //**********************************************************************************************

    pub fn spec(&mut self, id: SpecId, used_locals: BTreeMap<Var, H::SingleType>) -> IR::NopLabel {
        let label = IR::NopLabel(format!("{}", id).into());
        assert!(self
            .spec_info
            .insert(id, (label.clone(), used_locals))
            .is_none());
        label
    }
}
