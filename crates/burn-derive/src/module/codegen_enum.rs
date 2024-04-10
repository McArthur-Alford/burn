use super::{codegen::ModuleCodegen, record_enum::EnumModuleRecordCodegen};
use crate::shared::enum_variant::{parse_variants, EnumVariant};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

pub(crate) struct EnumModuleCodegen {
    pub variants: Vec<EnumVariant>,
}

impl ModuleCodegen for EnumModuleCodegen {
    type RecordCodegen = EnumModuleRecordCodegen;

    fn gen_num_params(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|_| {
            quote! {
                burn::module::Module::<B>::num_params(module)
            }
        });

        quote! {
            fn num_params(&self) -> usize {
                #match_body
            }
        }
    }

    fn gen_visit(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|_| {
            quote! {
                burn::module::Module::visit(module, visitor)
            }
        });

        quote! {
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
                #match_body
            }
        }
    }

    fn gen_collect_devices(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|_| {
            quote! {
                burn::module::Module::<B>::collect_devices(module, devices)
            }
        });

        quote! {
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>
            ) -> burn::module::Devices<B> {
                #match_body
            }
        }
    }

    fn gen_to_device(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(burn::module::Module::<B>::to_device(module, device))
            }
        });

        quote! {
            fn to_device(self, device: &B::Device) -> Self {
                #match_body
            }
        }
    }

    fn gen_fork(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(burn::module::Module::<B>::fork(module, device))
            }
        });

        quote! {
            fn fork(self, device: &B::Device) -> Self {
                #match_body
            }
        }
    }

    fn gen_map(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(burn::module::Module::<B>::map(module, mapper))
            }
        });

        quote! {
            fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
                #match_body
            }
        }
    }

    fn gen_valid(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::InnerModule::#variant(burn::module::AutodiffModule::<B>::valid(module))
            }
        });

        quote! {
            fn valid(&self) -> Self::InnerModule {
                #match_body
            }
        }
    }

    fn gen_into_record(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::Record::#variant(burn::module::Module::<B>::into_record(module))
            }
        });

        quote! {
            fn into_record(self) -> Self::Record {
                #match_body
            }
        }
    }

    fn gen_load_record(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                {
                    let Self::Record::#variant(r) = record else {panic!("Can't parse record from a different variant");};
                    Self::#variant(burn::module::Module::<B>::load_record(module, r))
                }
            }
        });

        quote! {
            fn load_record(self, record: Self::Record) -> Self {
                #match_body
            }
        }
    }

    fn gen_clone(&self) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant| {
            quote! {
                Self::#variant(module.clone())
            }
        });

        quote! {
            fn clone(&self) -> Self {
                #match_body
            }
        }
    }

    fn record_codegen(self) -> Self::RecordCodegen {
        EnumModuleRecordCodegen::new(self.variants)
    }

    fn gen_display(&self) -> TokenStream {
        quote! {
            fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
                self.fmt_single(fmt)?;
                write!(fmt, "\n")?;
                self.fmt_tree(fmt, 1)
            }
        }
    }

    fn gen_fmt_single(&self, name: &Ident) -> TokenStream {
        let match_body = self.gen_variants_match_fn(|variant: Ident| {
            quote! {
                write!(
                    fmt,
                    "{}::{} [num_params={}]",
                    stringify!(#name),
                    stringify!(#variant),
                    burn::module::Module::<B>::num_params(module)
                )
            }
        });

        quote! {
            fn fmt_single(&self, fmt: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
                #match_body
            }
        }
    }

    fn gen_fmt_tree(&self) -> TokenStream {
        let func = |name: Ident| {
            quote! {
                {
                    // Write the variant name and recurse
                    write!(fmt, "{}{}: ", "\t".repeat(depth), stringify!(#name))?;
                    burn::module::Module::<B>::fmt_single(module, fmt)?;
                    write!(fmt, "\n")?;
                    burn::module::Module::<B>::fmt_tree(module, fmt, depth + 1)?;
                }
            }
        };

        let match_body = self.gen_variants_match_fn(func);

        quote! {
            fn fmt_tree(&self, fmt: &mut core::fmt::Formatter<'_>, depth: usize) -> Result<(), core::fmt::Error> {
                #match_body
                Ok(())
            }
        }
    }
}

impl EnumModuleCodegen {
    pub fn from_ast(ast: &syn::DeriveInput) -> Self {
        Self {
            variants: parse_variants(ast),
        }
    }

    /// Generate the enum variants' match arm with the provided function
    fn gen_variants_match_fn<F>(&self, func: F) -> TokenStream
    where
        F: Fn(Ident) -> TokenStream,
    {
        let mut match_arms = quote! {};

        for variant in self.variants.iter() {
            let name = &variant.ident;
            let arm_pattern = quote! {Self::#name(module)};
            let arm_code = func(name.clone());

            match_arms.extend(quote! {#arm_pattern => #arm_code,})
        }

        quote! {
            match self {
                #match_arms
            }
        }
    }
}
