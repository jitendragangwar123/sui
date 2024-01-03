// Copyright (c) The Move Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::{
    diag,
    diagnostics::or_list_string,
    // diag,
    expansion::ast::{Fields, ModuleIdent, Value, Value_},
    hlir::translate::Context,
    naming::ast::{self as N, BuiltinTypeName_, Type, Var},
    parser::ast::{BinOp_, DatatypeName, Field, VariantName},
    shared::{
        ast_debug::{AstDebug, AstWriter},
        unique_map::UniqueMap,
    },
    typing::ast::{self as T, MatchArm_, MatchPattern, UnannotatedPat_ as TP},
};
use move_ir_types::location::*;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::Display,
};

//**************************************************************************************************
// Description
//**************************************************************************************************
// This mostly follows the classical Maranget (2008) implementation toward optimal decision trees.

//**************************************************************************************************
// Match Trees
//**************************************************************************************************

type PatBindings = BTreeMap<Var, T::Exp>;
type Guard = Option<(Vec<Var>, Box<T::Exp>)>;

#[derive(Clone, Debug)]
struct PatternArm {
    pat: VecDeque<T::MatchPattern>,
    guard: Guard,
    bindings: PatBindings,
    arm: usize,
}

#[derive(Clone, Debug)]
struct PatternMatrix {
    tys: Vec<Type>,
    patterns: Vec<PatternArm>,
}

#[derive(Clone, Debug)]
struct ArmResult {
    guard: Option<(Vec<Var>, Box<T::Exp>)>,
    bindings: PatBindings,
    arm: usize,
}

impl PatternArm {
    fn pattern_empty(&self) -> bool {
        self.pat.is_empty()
    }

    fn all_wild_arm(&mut self, fringe: &VecDeque<T::Exp>) -> Option<ArmResult> {
        self.push_bindings(fringe);
        if self
            .pat
            .iter()
            .all(|pat| matches!(pat.pat.value, T::UnannotatedPat_::Wildcard))
        {
            let PatternArm {
                pat: _,
                guard,
                bindings,
                arm,
            } = self;
            let arm = ArmResult {
                guard: guard.clone(),
                bindings: bindings.clone(),
                arm: *arm,
            };
            Some(arm)
        } else {
            None
        }
    }

    fn push_bindings(&mut self, fringe: &VecDeque<T::Exp>) {
        for (pmut, subject) in self.pat.iter_mut().zip(fringe.iter()) {
            if let TP::Binder(x) = pmut.pat.value {
                if let Some(_) = self.bindings.insert(x, subject.clone()) {
                    panic!("ICE should have failed in naming");
                };
                pmut.pat.value = TP::Wildcard;
            }
        }
    }

    fn first_ctor(&self) -> BTreeMap<VariantName, Fields<Type>> {
        if self.pat.is_empty() {
            return BTreeMap::new();
        }
        let mut names = BTreeMap::new();
        let mut ctor_queue = vec![self.pat.front().unwrap().clone()];
        while let Some(pat) = ctor_queue.pop() {
            match pat.pat.value {
                TP::Constructor(_, _, name, _, fields) => {
                    let ty_fields: Fields<Type> = fields.clone().map(|_, (ndx, (ty, _))| (ndx, ty));
                    names.insert(name, ty_fields);
                }
                TP::BorrowConstructor(_, _, name, _, fields) => {
                    let ty_fields: Fields<Type> = fields.clone().map(|_, (ndx, (ty, _))| (ndx, ty));
                    names.insert(name, ty_fields);
                }
                TP::Binder(_) => (),
                TP::Literal(_) => (),
                TP::Wildcard => (),
                TP::Or(lhs, rhs) => {
                    ctor_queue.push(*lhs);
                    ctor_queue.push(*rhs);
                }
                TP::At(_, inner) => {
                    ctor_queue.push(*inner);
                }
                TP::ErrorPat => (),
            }
        }
        names
    }

    fn first_lit(&self) -> BTreeSet<Value> {
        if self.pat.is_empty() {
            return BTreeSet::new();
        }
        let mut values = BTreeSet::new();
        let mut ctor_queue = vec![self.pat.front().unwrap().clone()];
        while let Some(pat) = ctor_queue.pop() {
            match pat.pat.value {
                TP::Constructor(_, _, _, _, _) => (),
                TP::BorrowConstructor(_, _, _, _, _) => (),
                TP::Binder(_) => (),
                TP::Literal(v) => {
                    values.insert(v);
                }
                TP::Wildcard => (),
                TP::Or(lhs, rhs) => {
                    ctor_queue.push(*lhs);
                    ctor_queue.push(*rhs);
                }
                TP::At(_, inner) => {
                    ctor_queue.push(*inner);
                }
                TP::ErrorPat => (),
            }
        }
        values
    }

    fn specialize(
        &self,
        context: &Context,
        subject: &T::Exp,
        ctor_name: &VariantName,
        arg_types: &Vec<&Type>,
    ) -> Option<PatternArm> {
        let mut output = self.clone();
        let first_pattern = output.pat.pop_front().unwrap();
        let loc = first_pattern.pat.loc;
        match first_pattern.pat.value {
            TP::Constructor(mident, enum_, name, _, fields)
            | TP::BorrowConstructor(mident, enum_, name, _, fields)
                if &name == ctor_name =>
            {
                let field_pats = fields.clone().map(|_key, (ndx, (_, pat))| (ndx, pat));
                let decl_fields = context.enum_variant_fields(&mident, &enum_, &name);
                let ordered_pats = order_fields_by_decl(decl_fields, field_pats);
                for (_, _, pat) in ordered_pats.into_iter().rev() {
                    output.pat.push_front(pat);
                }
                Some(output)
            }
            TP::Constructor(_, _, _, _, _) | TP::BorrowConstructor(_, _, _, _, _) => None,
            TP::Literal(_) => None,
            TP::Binder(_) => {
                for arg_type in arg_types
                    .clone()
                    .into_iter()
                    .map(|ty| ty_to_wildcard_pattern(ty.clone(), loc))
                    .rev()
                {
                    output.pat.push_front(arg_type);
                }
                Some(output)
            }
            TP::Wildcard => {
                for arg_type in arg_types
                    .clone()
                    .into_iter()
                    .map(|ty| ty_to_wildcard_pattern(ty.clone(), loc))
                    .rev()
                {
                    output.pat.push_front(arg_type);
                }
                Some(output)
            }
            TP::Or(_, _) => unreachable!(),
            TP::At(x, inner) => {
                output.pat.push_front(*inner);
                let inner_spec = output.specialize(context, subject, ctor_name, arg_types);
                match inner_spec {
                    None => None,
                    Some(mut inner) => {
                        if let Some(_) = inner.bindings.insert(x, subject.clone()) {
                            panic!("ICE should have failed in naming");
                        }
                        Some(inner)
                    }
                }
            }
            TP::ErrorPat => None,
        }
    }

    fn specialize_literal(&self, subject: &T::Exp, literal: &Value) -> Option<PatternArm> {
        let mut output = self.clone();
        let first_pattern = output.pat.pop_front().unwrap();
        match first_pattern.pat.value {
            TP::Literal(v) if &v == literal => Some(output),
            TP::Literal(_) => None,
            TP::Constructor(_, _, _, _, _) | TP::BorrowConstructor(_, _, _, _, _) => None,
            TP::Binder(_) => Some(output),
            TP::Wildcard => Some(output),
            TP::Or(_, _) => unreachable!(),
            TP::At(x, inner) => {
                output.pat.push_front(*inner);
                let inner_spec = output.specialize_literal(subject, literal);
                match inner_spec {
                    None => None,
                    Some(mut inner) => {
                        if let Some(_) = inner.bindings.insert(x, subject.clone()) {
                            panic!("ICE should have failed in naming");
                        }
                        Some(inner)
                    }
                }
            }
            TP::ErrorPat => None,
        }
    }

    fn default(&self, subject: &T::Exp) -> Option<PatternArm> {
        let mut output = self.clone();
        let first_pattern = output.pat.pop_front().unwrap();
        // let binder_opt = find_binder_opt(&first_pattern);

        match first_pattern.pat.value {
            TP::Literal(_) => None,
            TP::Constructor(_, _, _, _, _) | TP::BorrowConstructor(_, _, _, _, _) => None,
            TP::Binder(x) => {
                        if let Some(_) = output.bindings.insert(x, subject.clone()) {
                            panic!("ICE should have failed in naming");
                        }
                Some(output)
            }
            TP::Wildcard => Some(output),
            TP::Or(_, _) => unreachable!(),
            TP::At(x, inner) => {
                output.pat.push_front(*inner);
                let inner_spec = output.default(subject);
                match inner_spec {
                    None => None,
                    Some(mut inner) => {
                        if let Some(_) = inner.bindings.insert(x, subject.clone()) {
                            panic!("ICE should have failed in naming");
                        }
                        Some(inner)
                    }
                }
            }
            TP::ErrorPat => None,
        }
    }
}

impl PatternMatrix {
    fn from(subject_ty: Type, arms: Vec<T::MatchArm>) -> (PatternMatrix, Vec<T::Exp>) {
        let tys = vec![subject_ty];
        let mut patterns = vec![];
        let mut rhss = vec![];
        for sp!(_, arm) in arms {
            let MatchArm_ {
                pattern,
                binders: _,
                guard,
                rhs,
            } = arm;
            // assert!(guard.as_ref().is_none(), "Guards are not supported");
            rhss.push(*rhs);
            let arm_ndx = rhss.len() - 1;
            // print!("processing pat:");
            // pattern.print_verbose();
            let new_patterns = flatten_or(pattern);
            // println!("after flattening or:");
            // for pat in &new_patterns {
            //     pat.print_verbose();
            // }
            for pat in new_patterns {
                patterns.push(PatternArm {
                    pat: VecDeque::from([pat]),
                    guard: guard.clone().map(|guard| (vec![], guard)),
                    bindings: BTreeMap::new(),
                    arm: arm_ndx,
                });
            }
        }
        (PatternMatrix { tys, patterns }, rhss)
    }

    fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    fn patterns_empty(&self) -> bool {
        !self.patterns.is_empty() && self.patterns.iter().all(|pat| pat.pattern_empty())
    }

    fn wild_arm_opt(&mut self, fringe: &VecDeque<T::Exp>) -> Option<Vec<ArmResult>> {
        // NB: If the first row is all wild, we need to collect _all_ wild rows that have guards
        // until we find one that does not.
        if let Some(arm) = self.patterns[0].all_wild_arm(fringe) {
            if arm.guard.is_none() {
                return Some(vec![arm]);
            }
            let mut result = vec![arm];
            for pat in self.patterns[1..].iter_mut() {
                if let Some(arm) = pat.all_wild_arm(fringe) {
                    let has_guard = arm.guard.is_some();
                    result.push(arm);
                    if !has_guard {
                        return Some(result);
                    }
                }
            }
            Some(result)
        } else {
            None
        }
    }

    fn specialize(
        &self,
        context: &Context,
        subject: &T::Exp,
        ctor_name: &VariantName,
        arg_types: Vec<&Type>,
    ) -> PatternMatrix {
        let mut patterns = vec![];
        for entry in &self.patterns {
            if let Some(arm) = entry.specialize(context, subject, ctor_name, &arg_types) {
                patterns.push(arm)
            }
        }
        let mut tys = arg_types.into_iter().cloned().collect::<Vec<_>>();
        let mut old_tys = self.tys.clone();
        old_tys.remove(0);
        tys.extend(&mut old_tys.into_iter());
        PatternMatrix { tys, patterns }
    }

    fn specialize_literal(&self, subject: &T::Exp, lit: &Value) -> PatternMatrix {
        let mut patterns = vec![];
        for entry in &self.patterns {
            if let Some(arm) = entry.specialize_literal(subject, lit) {
                patterns.push(arm)
            }
        }
        let mut tys = self.tys.clone();
        tys.remove(0);
        PatternMatrix { tys, patterns }
    }

    fn default(&self, subject: &T::Exp) -> PatternMatrix {
        let mut patterns = vec![];
        for entry in &self.patterns {
            if let Some(arm) = entry.default(subject) {
                patterns.push(arm)
            }
        }
        let mut tys = self.tys.clone();
        tys.remove(0);
        PatternMatrix { tys, patterns }
    }

    fn first_head_ctors(&self) -> BTreeMap<VariantName, Fields<Type>> {
        let mut ctors = BTreeMap::new();
        for pat in &self.patterns {
            ctors.append(&mut pat.first_ctor());
        }
        ctors
    }

    fn first_lits(&self) -> BTreeSet<Value> {
        let mut ctors = BTreeSet::new();
        for pat in &self.patterns {
            ctors.append(&mut pat.first_lit());
        }
        ctors
    }

    fn has_guards(&self) -> bool {
        self.patterns.iter().any(|pat| pat.guard.is_some())
    }

    fn remove_guards(&mut self) {
        let pats = std::mem::take(&mut self.patterns);
        self.patterns = pats.into_iter().filter(|pat| pat.guard.is_none()).collect();
    }
}

fn ty_to_wildcard_pattern(ty: Type, loc: Loc) -> T::MatchPattern {
    T::MatchPattern {
        ty,
        pat: sp(loc, T::UnannotatedPat_::Wildcard),
    }
}

fn flatten_or(pat: MatchPattern) -> Vec<MatchPattern> {
    if matches!(
        pat.pat.value,
        TP::Literal(_) | TP::Binder(_) | TP::Wildcard | TP::ErrorPat
    ) {
        vec![pat]
    } else if matches!(
    &pat.pat.value,
    TP::Constructor(_, _, _, _, pats) | TP::BorrowConstructor(_, _, _, _, pats)
        if pats.is_empty()
    ) {
        return vec![pat];
    } else {
        let MatchPattern {
            ty,
            pat: sp!(ploc, pat_),
        } = pat;
        match pat_ {
            TP::Constructor(m, e, v, ta, spats) => {
                let all_spats = spats.map(|_, (ndx, (t, pat))| (ndx, (t, flatten_or(pat))));
                let fields_lists: Vec<Fields<(Type, MatchPattern)>> =
                    combine_pattern_fields(all_spats);
                fields_lists
                    .into_iter()
                    .map(|field_list| MatchPattern {
                        ty: ty.clone(),
                        pat: sp(ploc, TP::Constructor(m, e, v, ta.clone(), field_list)),
                    })
                    .collect::<Vec<_>>()
            }
            TP::BorrowConstructor(m, e, v, ta, spats) => {
                let all_spats = spats.map(|_, (ndx, (t, pat))| (ndx, (t, flatten_or(pat))));
                let fields_lists: Vec<Fields<(Type, MatchPattern)>> =
                    combine_pattern_fields(all_spats);
                fields_lists
                    .into_iter()
                    .map(|field_list| MatchPattern {
                        ty: ty.clone(),
                        pat: sp(ploc, TP::BorrowConstructor(m, e, v, ta.clone(), field_list)),
                    })
                    .collect::<Vec<_>>()
            }
            TP::Or(lhs, rhs) => {
                let mut lhs_rec = flatten_or(*lhs);
                let mut rhs_rec = flatten_or(*rhs);
                lhs_rec.append(&mut rhs_rec);
                lhs_rec
            }
            TP::At(x, inner) => flatten_or(*inner)
                .into_iter()
                .map(|pat| MatchPattern {
                    ty: ty.clone(),
                    pat: sp(ploc, TP::At(x, Box::new(pat))),
                })
                .collect::<Vec<_>>(),
            TP::Literal(_) | TP::Binder(_) | TP::Wildcard | TP::ErrorPat => unreachable!(),
        }
    }
}

fn combine_pattern_fields(
    fields: Fields<(Type, Vec<MatchPattern>)>,
) -> Vec<Fields<(Type, MatchPattern)>> {
    type VFields = Vec<(Field, (usize, (Spanned<N::Type_>, MatchPattern)))>;
    type VVFields = Vec<(Field, (usize, (Spanned<N::Type_>, Vec<MatchPattern>)))>;

    fn combine_recur(vec: &mut VVFields) -> Vec<VFields> {
        if let Some((f, (ndx, (ty, pats)))) = vec.pop() {
            let rec_fields = combine_recur(vec);
            // println!("rec fields: {:?}", rec_fields);
            let mut output = vec![];
            for entry in rec_fields {
                for pat in pats.clone() {
                    let mut entry = entry.clone();
                    entry.push((f, (ndx, (ty.clone(), pat))));
                    output.push(entry);
                }
            }
            // println!("output: {:?}", output);
            output
        } else {
            // Base case: a single match of no fields. We must have at least one, or else we would
            // not have called `combine_match_patterns`.
            vec![vec![]]
        }
    }

    fn vfields_to_fields(vfields: VFields) -> Fields<(Type, MatchPattern)> {
        UniqueMap::maybe_from_iter(vfields.into_iter()).unwrap()
    }

    // println!("init fields: {:?}", fields);
    let mut vvfields: VVFields = fields.into_iter().collect::<Vec<_>>();
    // println!("vv fields: {:?}", vvfields);
    let output_vec = combine_recur(&mut vvfields);
    // println!("output: {:?}", output_vec);
    output_vec
        .into_iter()
        .map(vfields_to_fields)
        .collect::<Vec<_>>()
}

//**************************************************************************************************
// Match Compilation
//**************************************************************************************************

type Fringe = VecDeque<T::Exp>;

enum MatchStep {
    Leaf(Vec<ArmResult>),
    Failure,
    LiteralSwitch(
        T::Exp,
        Fringe,
        BTreeMap<Value, PatternMatrix>,
        PatternMatrix,
    ),
    VariantSwitch(
        T::Exp,
        Vec<Type>,
        BTreeMap<VariantName, (Option<Vec<(Field, Var, Type)>>, Fringe, PatternMatrix)>,
        (Fringe, PatternMatrix),
    ),
}

#[derive(Clone)]
enum WorkResult {
    Leaf(Vec<ArmResult>),
    Failure,
    LiteralSwitch(
        T::Exp,
        BTreeMap<Value, usize>,
        usize, // default
    ),
    VariantSwitch(
        T::Exp,
        Vec<Type>,
        BTreeMap<VariantName, (Option<Vec<(Field, Var, Type)>>, usize)>,
        usize,
    ),
}

pub fn compile_match(
    context: &mut Context,
    result_type: &Type,
    subject: T::Exp,
    arms: Spanned<Vec<T::MatchArm>>,
) -> T::Exp {
    let arms_loc = arms.loc;
    let (pattern_matrix, arms) = PatternMatrix::from(subject.ty.clone(), arms.value);

    let mut counterexample_matrix = pattern_matrix.clone();
    let has_guards = counterexample_matrix.has_guards();
    counterexample_matrix.remove_guards();
    if find_counterexample(context, subject.exp.loc, counterexample_matrix, has_guards) {
        return T::exp(
            result_type.clone(),
            sp(subject.exp.loc, T::UnannotatedExp_::UnresolvedError),
        );
    }

    let mut compilation_results: BTreeMap<usize, WorkResult> = BTreeMap::new();

    let (initial_binder, subject) = {
        let subject_loc = subject.exp.loc;
        let subject_ty = subject.ty.clone();
        let var = context.new_naming_temp(subject_loc);
        let rhs = subject;
        let initial_binder = make_binding(var, rhs.ty.clone(), rhs);

        let subject_exp = T::UnannotatedExp_::Move {
            from_user: false,
            var,
        };
        let subject = T::exp(subject_ty, sp(subject_loc, subject_exp));

        (initial_binder, subject)
    };

    let mut work_queue: Vec<(usize, Fringe, PatternMatrix)> =
        vec![(0, VecDeque::from([subject]), pattern_matrix)];

    let mut work_id = 0;

    let mut next_id = || {
        work_id += 1;
        work_id
    };

    while let Some((cur_id, init_fringe, matrix)) = work_queue.pop() {
        // println!("---\nwork queue entry: {}", cur_id);
        // println!("fringe:");
        // for elem in &init_fringe {
        //     print!("  ");
        //     elem.print_verbose();
        // }
        // println!("matrix:");
        // matrix.print_verbose();
        let redefined: Option<WorkResult> =
            match compile_match_head(context, init_fringe.clone(), matrix) {
                MatchStep::Leaf(leaf) => compilation_results.insert(cur_id, WorkResult::Leaf(leaf)),
                MatchStep::Failure => compilation_results.insert(cur_id, WorkResult::Failure),
                // MatchStep::Bind {
                //     source,
                //     target,
                //     ty,
                //     fringe,
                //     matrix,
                // } => {
                //     let work_id = next_id();
                //     work_queue.push((work_id, fringe, matrix));
                //     compilation_results
                //         .insert(cur_id, WorkResult::Bind(source, target, ty, work_id))
                // }
                MatchStep::LiteralSwitch(subject, fringe, arms, default) => {
                    let mut answer_map = BTreeMap::new();
                    for (value, matrix) in arms {
                        let work_id = next_id();
                        answer_map.insert(value, work_id);
                        work_queue.push((work_id, fringe.clone(), matrix));
                    }
                    let default_work_id = next_id();
                    work_queue.push((default_work_id, fringe, default));
                    compilation_results.insert(
                        cur_id,
                        WorkResult::LiteralSwitch(subject, answer_map, default_work_id),
                    )
                }
                MatchStep::VariantSwitch(subject, tyargs, variants, (dfringe, dmatrix)) => {
                    let mut answer_map = BTreeMap::new();
                    for (name, (dtor_fields, fringe, matrix)) in variants {
                        let work_id = next_id();
                        answer_map.insert(name, (dtor_fields, work_id));
                        work_queue.push((work_id, fringe, matrix));
                    }
                    let default_work_id = next_id();
                    work_queue.push((default_work_id, dfringe, dmatrix));
                    compilation_results.insert(
                        cur_id,
                        WorkResult::VariantSwitch(subject, tyargs, answer_map, default_work_id),
                    )
                }
            };
        assert!(redefined.is_none(), "ICE match work queue went awry");
    }

    let match_start = compilation_results.remove(&0).unwrap();
    let mut resolution_context = ResolutionContext {
        hlir_context: context,
        output_type: result_type,
        arms: &arms,
        arms_loc,
        results: &mut compilation_results,
    };
    let match_exp = resolve_result(&mut resolution_context, match_start);

    let eloc = match_exp.exp.loc;
    let mut seq = VecDeque::new();
    seq.push_back(initial_binder);
    seq.push_back(sp(eloc, T::SequenceItem_::Seq(Box::new(match_exp))));
    let exp_value = sp(eloc, T::UnannotatedExp_::Block(seq));
    T::exp(result_type.clone(), exp_value)
}

fn compile_match_head(
    context: &mut Context,
    mut fringe: VecDeque<T::Exp>,
    mut matrix: PatternMatrix,
) -> MatchStep {
    if matrix.is_empty() {
        MatchStep::Failure
    } else if let Some(leaf) = matrix.wild_arm_opt(&fringe) {
        MatchStep::Leaf(leaf)
    } else if fringe[0].ty.value.unfold_to_builtin_type_name().is_some() {
        let subject = fringe
            .pop_front()
            .expect("ICE empty fringe in match compilation");
        // treat column as a literal
        let lits = matrix.first_lits();
        let mut arms = BTreeMap::new();
        for lit in lits {
            let inner_matrix = matrix.specialize_literal(&subject, &lit);
            assert!(arms.insert(lit, inner_matrix).is_none());
        }
        let default = matrix.default(&subject);
        MatchStep::LiteralSwitch(subject, fringe, arms, default)
    } else {
        let subject = fringe
            .pop_front()
            .expect("ICE empty fringe in match compilation");
        // println!("------\nsubject:");
        // subject.print_verbose();
        // println!("--\ncompile match head:");
        // subject.print_verbose();
        // println!("--\nmatrix;");
        // matrix.print_verbose();

        let (mident, enum_name) = subject
            .ty
            .value
            .unfold_to_type_name()
            .and_then(|sp!(_, name)| name.datatype_name())
            .expect("ICE non-datatype type in head constructor fringe position");
        let tyargs = subject.ty.value.type_arguments().unwrap().clone();
        // treat it as a head constructor
        // assert!(!ctors.is_empty());

        let mut unmatched_variants = context
            .enum_variants(&mident, &enum_name)
            .into_iter()
            .collect::<BTreeSet<_>>();

        let ctors = matrix.first_head_ctors();

        let mut arms = BTreeMap::new();
        for (ctor, arg_types) in ctors {
            unmatched_variants.remove(&ctor);
            let fringe_binders = context.make_match_binders(arg_types);
            let fringe_exps = make_fringe_exps(&fringe_binders);
            let mut inner_fringe = fringe.clone();
            for fringe_exp in fringe_exps.into_iter().rev() {
                inner_fringe.push_front(fringe_exp);
            }
            let bind_tys = fringe_binders
                .iter()
                .map(|(_, _, ty)| ty)
                .collect::<Vec<_>>();
            let inner_matrix = matrix.specialize(context, &subject, &ctor, bind_tys);
            // println!("specializing to {:?}", ctor);
            // print!("specialized:");
            // inner_matrix.print_verbose();
            assert!(arms
                .insert(ctor, (Some(fringe_binders), inner_fringe, inner_matrix))
                .is_none());
        }

        let default_matrix = matrix.default(&subject);

        MatchStep::VariantSwitch(subject, tyargs, arms, (fringe, default_matrix))
    }
}

pub fn order_fields_by_decl<T>(
    decl_fields: Option<&UniqueMap<Field, usize>>,
    fields: Fields<T>,
) -> Vec<(usize, Field, T)> {
    let mut texp_fields: Vec<(usize, Field, T)> = if let Some(field_map) = decl_fields {
        fields
            .into_iter()
            .map(|(f, (_exp_idx, t))| (*field_map.get(&f).unwrap(), f, t))
            .collect()
    } else {
        // If no field map, compiler error in typing.
        fields
            .into_iter()
            .enumerate()
            .map(|(ndx, (f, (_exp_idx, t)))| (ndx, f, t))
            .collect()
    };
    texp_fields.sort_by(|(decl_idx1, _, _), (decl_idx2, _, _)| decl_idx1.cmp(decl_idx2));
    texp_fields
}

fn make_fringe_exps(binders: &[(Field, Var, Type)]) -> VecDeque<T::Exp> {
    binders
        .iter()
        .map(|(_, x, t)| T::Exp {
            exp: sp(
                t.loc,
                T::UnannotatedExp_::Move {
                    from_user: false,
                    var: *x,
                },
            ),
            ty: t.clone(),
        })
        .collect::<VecDeque<_>>()
}

//------------------------------------------------
// Result Construction
//------------------------------------------------

struct ResolutionContext<'ctxt, 'call> {
    hlir_context: &'call mut Context<'ctxt>,
    output_type: &'call Type,
    arms: &'call Vec<T::Exp>,
    arms_loc: Loc,
    results: &'call mut BTreeMap<usize, WorkResult>,
}

impl<'ctxt, 'call> ResolutionContext<'ctxt, 'call> {
    fn arm(&self, index: usize) -> T::Exp {
        self.arms[index].clone()
    }

    fn arms_loc(&self) -> Loc {
        self.arms_loc
    }

    fn work_result(&mut self, work_id: usize) -> WorkResult {
        self.results.remove(&work_id).unwrap()
    }

    fn copy_work_result(&mut self, work_id: usize) -> WorkResult {
        self.results.get(&work_id).unwrap().clone()
    }

    fn output_type(&self) -> Type {
        self.output_type.clone()
    }
}

fn resolve_result(context: &mut ResolutionContext, result: WorkResult) -> T::Exp {
    match result {
        WorkResult::Leaf(mut leaf) => make_leaf(context, &mut leaf),
        WorkResult::Failure => T::exp(
            context.output_type(),
            sp(context.arms_loc, T::UnannotatedExp_::UnresolvedError),
        ),
        WorkResult::VariantSwitch(subject, tyargs, mut entries, default_ndx) => {
            let (m, e) = subject
                .ty
                .value
                .unfold_to_type_name()
                .and_then(|sp!(_, name)| name.datatype_name())
                .unwrap();

            let sorted_variants: Vec<VariantName> = context.hlir_context.enum_variants(&m, &e);
            let mut blocks = vec![];
            for v in sorted_variants {
                if let Some((unpack_fields, result_ndx)) = entries.remove(&v) {
                    let work_result = context.work_result(result_ndx);
                    let rest_result = resolve_result(context, work_result);
                    let unpack_block = make_unpack(
                        context.hlir_context,
                        m,
                        e,
                        v,
                        tyargs.clone(),
                        unpack_fields,
                        subject.clone(),
                        rest_result,
                    );
                    blocks.push((v, unpack_block));
                } else {
                    let work_result = context.copy_work_result(default_ndx);
                    let rest_result = resolve_result(context, work_result);
                    blocks.push((v, rest_result));
                }
            }
            let out_exp =
                T::UnannotatedExp_::VariantMatch(make_var_ref(subject), e, blocks);
            T::exp(context.output_type(), sp(context.arms_loc(), out_exp))
        }
        WorkResult::LiteralSwitch(exp, mut map, _defaulkt)
            if matches!(
                exp.ty.value.builtin_name(),
                Some(sp!(_, BuiltinTypeName_::Bool))
            ) && map.len() == 2 =>
        {
            // If the literal switch for a boolean is saturated, no default case.
            let lit_subject = make_lit_copy(exp.clone());
            let true_arm_ndx = map.remove(&sp(Loc::invalid(), Value_::Bool(true))).unwrap();
            let false_arm_ndx = map
                .remove(&sp(Loc::invalid(), Value_::Bool(false)))
                .unwrap();

            let true_arm_result = context.work_result(true_arm_ndx);
            let false_arm_result = context.work_result(false_arm_ndx);

            let true_arm = resolve_result(context, true_arm_result);
            let false_arm = resolve_result(context, false_arm_result);
            let result_type = true_arm.ty.clone();

            make_if_else(lit_subject, true_arm, false_arm, result_type)
        }
        WorkResult::LiteralSwitch(exp, map, default) => {
            let lit_subject = make_lit_copy(exp.clone());

            let mut entries = map.into_iter().collect::<Vec<_>>();
            entries.sort_by(|(key1, _), (key2, _)| key1.cmp(key2));

            let else_work_result = context.work_result(default);
            let mut out_exp = resolve_result(context, else_work_result);

            for (key, result_ndx) in entries.into_iter().rev() {
                let work_result = context.work_result(result_ndx);
                let match_arm = resolve_result(context, work_result);
                let test_exp = make_lit_test(lit_subject.clone(), key);
                let result_ty = out_exp.ty.clone();
                out_exp = make_if_else(test_exp, match_arm, out_exp, result_ty);
            }
            out_exp
        }
    }
}

fn make_leaf(context: &ResolutionContext, leaf: &mut Vec<ArmResult>) -> T::Exp {
    assert!(!leaf.is_empty(), "ICE empty leaf in matching");
    let last = leaf.pop().unwrap();
    assert!(last.guard.is_none(), "ICE must have a non-guarded leaf");
    let mut out_exp = make_bindings(last.bindings, context.arm(last.arm));
    let out_ty = out_exp.ty.clone();
    while let Some(arm) = leaf.pop() {
        assert!(arm.guard.is_some(), "ICE expected a guard");
        out_exp = make_guard_exp(context, arm, out_exp, out_ty.clone());
    }
    out_exp
}

fn make_guard_exp(context: &ResolutionContext, arm: ArmResult, cur_exp: T::Exp, result_ty: Type) -> T::Exp {
    let ArmResult { guard, bindings, arm } = arm;
    let (guard_vars, guard_test) = guard.unwrap();
    let mut exp_map = BTreeMap::new();

    for (var, exp) in bindings.clone() {
        exp_map.insert(var, make_var_ref(exp));
    }

    let mut guard_bindings = BTreeMap::new();
    for var in guard_vars {
        let var_exp = bindings.get(&var).expect("ICE should have failed in naming");
        guard_bindings.insert(var, *make_var_ref(var_exp.clone()));
    }
    let guard_arm = make_bindings(bindings, context.arm(arm));
    let body = make_if_else(*guard_test, guard_arm, cur_exp, result_ty);
    make_bindings(guard_bindings, body)
}

fn make_var_ref(subject: T::Exp) -> Box<T::Exp> {
    let T::Exp { ty, exp } = subject;
    match ty {
        sp!(_, N::Type_::Ref(false, _)) => {
            let loc = exp.loc;
            let var = match exp.value {
                T::UnannotatedExp_::Move { var, from_user: _ } => var,
                _ => panic!("ICE Non-var in match fringe [imm-ref case]"),
            };
            Box::new(make_copy_exp(ty, loc, var))
        }
        sp!(_, N::Type_::Ref(true, inner)) => {
            let loc = exp.loc;
            let var = match exp.value {
                T::UnannotatedExp_::Move { var, from_user: _ } => var,
                _ => panic!("ICE Non-var in match fringe [mut-ref case]"),
            };

            // NB(cswords): we now freeze the mut ref at the non-mut ref type.

            let ref_ty = sp(loc, N::Type_::Ref(true, inner.clone()));
            let freeze_arg = make_copy_exp(ref_ty, loc, var);
            let freeze_ty = sp(loc, N::Type_::Ref(false, inner));
            Box::new(make_freeze_exp(freeze_ty, loc, freeze_arg))
        }
        ty => {
            let loc = exp.loc;
            let var = match exp.value {
                T::UnannotatedExp_::Move { var, from_user: _ } => var,
                _ => panic!("ICE Non-var in match fringe [value case]"),
            };
            let ref_ty = sp(loc, N::Type_::Ref(false, Box::new(ty)));
            let borrow_exp = T::UnannotatedExp_::BorrowLocal(false, var);
            Box::new(T::exp(ref_ty, sp(loc, borrow_exp)))
        }
    }
}

fn make_lit_copy(subject: T::Exp) -> T::Exp {
    let T::Exp { ty, exp } = subject;
    match ty {
        sp!(ty_loc, N::Type_::Ref(false, inner)) => {
            let loc = exp.loc;
            let var = match exp.value {
                T::UnannotatedExp_::Move { var, from_user: _ } => var,
                _ => panic!("ICE Non-var in match fringe [imm-ref case]"),
            };
            let copy_exp = make_copy_exp(sp(ty_loc, N::Type_::Ref(false, inner.clone())), loc, var);
            make_deref_exp(*inner, loc, copy_exp)
        }
        sp!(_, N::Type_::Ref(true, inner)) => {
            let loc = exp.loc;
            let var = match exp.value {
                T::UnannotatedExp_::Move { var, from_user: _ } => var,
                _ => panic!("ICE Non-var in match fringe [mut-ref case]"),
            };

            // NB(cswords): we now freeze the mut ref at the non-mut ref type.
            let ref_ty = sp(loc, N::Type_::Ref(true, inner.clone()));
            let freeze_arg = make_copy_exp(ref_ty, loc, var);
            let freeze_ty = sp(loc, N::Type_::Ref(false, inner.clone()));
            let frozen_exp = make_freeze_exp(freeze_ty, loc, freeze_arg);
            make_deref_exp(*inner, loc, frozen_exp)
        }
        ty => {
            let loc = exp.loc;
            let var = match exp.value {
                T::UnannotatedExp_::Move { var, from_user: _ } => var,
                _ => panic!("ICE Non-var in match fringe [value case]"),
            };
            make_copy_exp(ty, loc, var)
        }
    }
}

fn make_copy_exp(ty: Type, loc: Loc, var: Var) -> T::Exp {
    let exp_ = T::UnannotatedExp_::Copy {
        var,
        from_user: false,
    };
    T::exp(ty, sp(loc, exp_))
}

fn make_freeze_exp(ty: Type, loc: Loc, arg: T::Exp) -> T::Exp {
    let freeze_fn = Box::new(sp(loc, T::BuiltinFunction_::Freeze(ty.clone())));
    let freeze_exp = T::UnannotatedExp_::Builtin(freeze_fn, Box::new(arg));
    T::exp(ty, sp(loc, freeze_exp))
}

fn make_deref_exp(ty: Type, loc: Loc, arg: T::Exp) -> T::Exp {
    let deref_exp = T::UnannotatedExp_::Dereference(Box::new(arg));
    T::exp(ty, sp(loc, deref_exp))
}

fn make_unpack(
    context: &mut Context,
    mident: ModuleIdent,
    enum_: DatatypeName,
    variant: VariantName,
    tyargs: Vec<Type>,
    unpack_fields: Option<Vec<(Field, Var, Type)>>,
    rhs: T::Exp,
    next: T::Exp,
) -> T::Exp {
    // println!("rhs for make_unpack");
    // rhs.print_verbose();
    if let Some(fields) = unpack_fields {
        // println!("unpacking fields for {variant}");
        let rhs_loc = rhs.exp.loc;
        let mut seq = VecDeque::new();

        let mut lvalue_fields: Fields<(Type, T::LValue)> = UniqueMap::new();

        for (ndx, (field_name, var, ty)) in fields.into_iter().enumerate() {
            let var_lvalue = make_lvalue(var, ty.clone());
            lvalue_fields
                .add(field_name, (ndx, (ty, var_lvalue)))
                .unwrap();
        }

        let unpack_lvalue = match rhs.ty.value {
            N::Type_::Ref(mut_, _) => sp(
                rhs_loc,
                T::LValue_::BorrowUnpackVariant(
                    mut_,
                    mident,
                    enum_,
                    variant,
                    tyargs,
                    lvalue_fields,
                ),
            ),
            _ => sp(
                rhs_loc,
                T::LValue_::UnpackVariant(mident, enum_, variant, tyargs, lvalue_fields),
            ),
        };
        // print!("binder ty: ");
        // rhs.ty.print_verbose();
        let binder = T::SequenceItem_::Bind(
            sp(rhs_loc, vec![unpack_lvalue]),
            vec![Some(rhs.ty.clone())],
            Box::new(rhs),
        );
        seq.push_back(sp(rhs_loc, binder));

        let result_type = next.ty.clone();
        let eloc = next.exp.loc;
        seq.push_back(sp(eloc, T::SequenceItem_::Seq(Box::new(next))));
        let exp_value = sp(eloc, T::UnannotatedExp_::Block(seq));
        T::exp(result_type, exp_value)
    } else {
        let var = context.new_naming_temp(rhs.exp.loc);
        make_bindings(BTreeMap::from([(var, rhs)]), next)
    }
}

fn make_bindings(bindings: PatBindings, next: T::Exp) -> T::Exp {
    let eloc = next.exp.loc;
    let mut seq = VecDeque::new();
    for (lhs, rhs) in bindings {
        seq.push_back(make_binding(lhs, rhs.ty.clone(), rhs));
    }
    let result_type = next.ty.clone();
    seq.push_back(sp(eloc, T::SequenceItem_::Seq(Box::new(next))));
    let exp_value = sp(eloc, T::UnannotatedExp_::Block(seq));
    T::exp(result_type, exp_value)
}

fn make_lvalue(lhs: Var, ty: Type) -> T::LValue {
    let lhs_loc = lhs.loc;
    let lhs_var = T::LValue_::Var {
        var: lhs,
        ty: Box::new(ty.clone()),
        unused_binding: false,
    };
    sp(lhs_loc, lhs_var)
}

fn make_binding(lhs: Var, ty: Type, rhs: T::Exp) -> T::SequenceItem {
    let lhs_loc = lhs.loc;
    let lhs_lvalue = make_lvalue(lhs, ty.clone());
    let binder =
        T::SequenceItem_::Bind(sp(lhs_loc, vec![lhs_lvalue]), vec![Some(ty)], Box::new(rhs));
    sp(lhs_loc, binder)
}

fn make_lit_test(lit_exp: T::Exp, value: Value) -> T::Exp {
    let loc = value.loc;
    let value_exp = Box::new(T::exp(
        lit_exp.ty.clone(),
        sp(loc, T::UnannotatedExp_::Value(value)),
    ));
    let bool = N::Type_::bool(loc);
    let equality_exp_ = T::UnannotatedExp_::BinopExp(
        Box::new(lit_exp),
        sp(loc, BinOp_::Eq),
        Box::new(bool.clone()),
        value_exp,
    );
    T::exp(bool, sp(loc, equality_exp_))
}

fn make_if_else(test: T::Exp, conseq: T::Exp, alt: T::Exp, result_ty: Type) -> T::Exp {
    // FIXME: this span is woefully wrong
    let loc = conseq.exp.loc;
    T::exp(
        result_ty,
        sp(
            loc,
            T::UnannotatedExp_::IfElse(Box::new(test), Box::new(conseq), Box::new(alt)),
        ),
    )
}

//------------------------------------------------
// Counterexample Generation
//------------------------------------------------

#[derive(Clone, Debug)]
enum CounterExample {
    Wildcard,
    Literal(String),
    Constructor(DatatypeName, VariantName, Vec<CounterExample>),
    Note(String, Box<CounterExample>),
}

impl CounterExample {
    fn into_notes(self) -> VecDeque<String> {
        match self {
            CounterExample::Wildcard => VecDeque::new(),
            CounterExample::Literal(_) => VecDeque::new(),
            CounterExample::Note(s, next) => {
                let mut notes = next.into_notes();
                notes.push_front(s.clone());
                notes
            }
            CounterExample::Constructor(_, _, inner) => inner
                .into_iter()
                .flat_map(|ce| ce.into_notes())
                .collect::<VecDeque<_>>(),
        }
    }
}

impl Display for CounterExample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CounterExample::Wildcard => write!(f, "_"),
            CounterExample::Literal(s) => write!(f, "{}", s),
            CounterExample::Note(_, inner) => inner.fmt(f),
            CounterExample::Constructor(e, v, args) => {
                write!(f, "{}::{}", e, v)?;
                if !args.is_empty() {
                    write!(f, "(")?;
                    write!(
                        f,
                        "{}",
                        args.iter()
                            .map(|arg| format!("{}", arg))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?;
                    write!(f, ")")
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Returns true if it found a counter-example.
fn find_counterexample(context: &mut Context, loc: Loc, matrix: PatternMatrix, has_guards: bool) -> bool {
    fn make_wildcards(n: usize) -> Vec<CounterExample> {
        std::iter::repeat(CounterExample::Wildcard)
            .take(n)
            .collect()
    }

    fn dummy_subject() -> T::Exp {
        T::exp(
            sp(Loc::invalid(), N::Type_::Anything),
            sp(Loc::invalid(), T::UnannotatedExp_::UnresolvedError),
        )
    }

    // \mathcal{I} from Maranget. Warning for pattern matching. 1992.
    fn find_counterexample(
        context: &mut Context,
        matrix: PatternMatrix,
        arity: u32,
        ndx: &mut u32,
    ) -> Option<Vec<CounterExample>> {
        // println!("checking matrix");
        // matrix.print_verbose();
        let result = if matrix.patterns_empty() {
            None
        } else if matrix.is_empty() {
            Some(make_wildcards(arity as usize))
        } else if let Some(sp!(_, BuiltinTypeName_::Bool)) =
            matrix.tys.first().unwrap().value.builtin_name()
        {
            let literals = matrix.first_lits();
            assert!(literals.len() <= 2, "ICE match exhaustiveness failure");
            if literals.len() == 2 {
                // Saturated
                for lit in literals {
                    if let Some(counterexample) = find_counterexample(
                        context,
                        matrix.specialize_literal(&dummy_subject(), &lit),
                        arity - 1,
                        ndx,
                    ) {
                        let lit_str = format!("{}", lit);
                        let result = [CounterExample::Literal(lit_str)]
                            .into_iter()
                            .chain(counterexample)
                            .collect();
                        return Some(result);
                    }
                }
                None
            } else {
                let default = matrix.default(&dummy_subject());
                if let Some(counterexample) = find_counterexample(context, default, arity - 1, ndx)
                {
                    if literals.is_empty() {
                        let result = [CounterExample::Wildcard]
                            .into_iter()
                            .chain(counterexample)
                            .collect();
                        Some(result)
                    } else {
                        let mut unused = BTreeSet::from([Value_::Bool(true), Value_::Bool(false)]);
                        for lit in literals {
                            unused.remove(&lit.value);
                        }
                        let result = [CounterExample::Literal(format!(
                            "{}",
                            unused.first().unwrap()
                        ))]
                        .into_iter()
                        .chain(counterexample)
                        .collect();
                        Some(result)
                    }
                } else {
                    None
                }
            }
        } else if let Some(sp!(_, _)) = matrix.tys[0].value.unfold_to_builtin_type_name() {
            // For all other non-literals, we don't consider a case where the constructors are
            // saturated.
            let literals = matrix.first_lits();
            let default = matrix.default(&dummy_subject());
            if let Some(counterexample) = find_counterexample(context, default, arity - 1, ndx) {
                if literals.is_empty() {
                    let result = [CounterExample::Wildcard]
                        .into_iter()
                        .chain(counterexample)
                        .collect();
                    Some(result)
                } else {
                    let n_id = format!("_{}", ndx);
                    *ndx += 1;
                    let lit_strs = literals
                        .into_iter()
                        .map(|lit| format!("{}", lit))
                        .collect::<Vec<_>>();
                    let lit_str = or_list_string(lit_strs);
                    let lit_msg = format!("When '{}' is not {}", n_id, lit_str);
                    let lit_ce =
                        CounterExample::Note(lit_msg, Box::new(CounterExample::Literal(n_id)));
                    let result = [lit_ce].into_iter().chain(counterexample).collect();
                    Some(result)
                }
            } else {
                None
            }
        } else {
            // println!("matrix types:");
            // for ty in &matrix.tys {
            //     ty.print_verbose();
            // }
            let (mident, enum_name) = matrix.tys[0]
                .value
                .unfold_to_type_name()
                .map(|name| {
                    println!("name: {:#?}", name);
                    name
                })
                .and_then(|sp!(_, name)| name.datatype_name())
                .expect("ICE non-datatype type in head constructor fringe position");

            let mut unmatched_variants = context
                .enum_variants(&mident, &enum_name)
                .into_iter()
                .collect::<BTreeSet<_>>();

            let ctors = matrix.first_head_ctors();
            for ctor in ctors.keys() {
                unmatched_variants.remove(ctor);
            }
            if unmatched_variants.is_empty() {
                for (ctor, arg_types) in ctors {
                    let ctor_arity = arg_types.len() as u32;
                    let fringe_binders = context.make_match_binders(arg_types);
                    let bind_tys = fringe_binders
                        .iter()
                        .map(|(_, _, ty)| ty)
                        .collect::<Vec<_>>();
                    let inner_matrix =
                        matrix.specialize(context, &dummy_subject(), &ctor, bind_tys);
                    if let Some(mut counterexample) =
                        find_counterexample(context, inner_matrix, ctor_arity + arity - 1, ndx)
                    {
                        let ctor_args = counterexample
                            .drain(0..(ctor_arity as usize))
                            .collect::<Vec<_>>();
                        let output = [CounterExample::Constructor(enum_name, ctor, ctor_args)]
                            .into_iter()
                            .chain(counterexample)
                            .collect();
                        return Some(output);
                    }
                }
                None
            } else {
                let default = matrix.default(&dummy_subject());
                if let Some(counterexample) = find_counterexample(context, default, arity - 1, ndx)
                {
                    if ctors.is_empty() {
                        // If we didn't match any head constructor, `_` is a reasonable
                        // counter-example entry.
                        let mut result = vec![CounterExample::Wildcard];
                        result.extend(&mut counterexample.into_iter());
                        Some(result)
                    } else {
                        let variant_name = unmatched_variants.first().unwrap();
                        let ctor_arity = context
                            .enum_variant_fields(&mident, &enum_name, variant_name)
                            .unwrap()
                            .iter()
                            .count();
                        let args = make_wildcards(ctor_arity);
                        let result = [CounterExample::Constructor(enum_name, *variant_name, args)]
                            .into_iter()
                            .chain(counterexample)
                            .collect();
                        Some(result)
                    }
                } else {
                    // If we are missing a variant but everything else is fine, we're done.
                    None
                }
            }
        };
        // print!("result:");
        // match result {
        //     Some(ref n) => println!("{:#?}", n),
        //     None => println!("NON"),
        // }
        // println!();
        result
    }

    // let result = fancy_i(context, matrix, 1);
    // match result {
    //     Some(ref n) => println!("{}", n[0]),
    //     None => println!("NON"),
    // }

    let mut ndx = 0;
    if let Some(mut counterexample) = find_counterexample(context, matrix, 1, &mut ndx) {
        // println!("counterexamples: {}", counterexample.len());
        // for ce in &counterexample {
        //     println!("{}", ce);
        // }
        assert!(counterexample.len() == 1);
        let counterexample = counterexample.remove(0);
        let msg = format!("Pattern '{}' not covered", counterexample);
        let mut diag = diag!(TypeSafety::IncompletePattern, (loc, msg));
        for note in counterexample.into_notes() {
            diag.add_note(note);
        }
        if has_guards {
            diag.add_note("Match arms with guards are not considered for coverage.");
        }
        context.env.add_diag(diag);
        true
    } else {
        false
    }
}

//**************************************************************************************************
// Debug Print
//**************************************************************************************************

impl AstDebug for PatternMatrix {
    fn ast_debug(&self, w: &mut AstWriter) {
        for arm in &self.patterns {
            let PatternArm {
                pat,
                guard,
                bindings: binders,
                arm,
            } = arm;
            w.comma(pat, |w, p| p.ast_debug(w));
            w.write(" =>");
            if let Some(guard) = guard {
                w.write(" if ");
                guard.1.ast_debug(w);
            }
            w.write(" [");
            w.comma(binders, |w, (x, e)| {
                x.ast_debug(w);
                w.write(" <- ");
                e.ast_debug(w);
            });
            w.write(format!("] arm {}\n", arm));
        }
    }
}
