
import os
import pickle
import tqdm
import logging
import glob
import itertools
import pvl

import pipeline
import helpers
import config
import database as db
# from models.image import VoyagerImage
from repository.image import create_user_from_dict, flatten_vicar_object, upsert_foobar


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db.Base.metadata.create_all(bind=db.engine)


# Steps I need 
# - Open all tar files, extract relevant metadata and images, cache them in the local fs
# - Extract and store metadata of interest 
# - Link everything related by an ID
# - Load images and cache them


class ListTarFiles(pipeline.Task):

    def __init__(self, name, source_path):

        super().__init__(name=name)
        self.source_path = source_path

    def process(self,item ):
        import glob

        for g in glob.iglob(f"{self.source_path}/*.tar.gz"):
            yield {
                "tar_file_path": g,
                "stem": helpers.extract_stem(g)
            }



class ExtractTarfile(pipeline.Task):

    def __init__(self, name, cache_path):

        super().__init__( name )

        self.cache_path = cache_path

        self.cached_files_list = self._get_cache_file_list()
        logger.info( f"cached files list has {len(self.cached_files_list)} entries" )

        # print( self.cached_files_list)
        # raise Exception("stop")

    def _parse_member_name(self, member_info ):
        import pathlib

        member_name = member_info.get_info()["name"]
        inner_p = pathlib.Path( member_name )
        inner_suffix = inner_p.suffix.replace(".","")
        inner_stem = inner_p.stem

        if "_" in inner_stem:
            image_id, image_type = inner_stem.split("_")[0], inner_stem.split("_")[1]
        else:
            image_id = None
            image_type = None

        return {
            "stem": inner_stem,
            "image_type": image_type,
            "suffix": inner_suffix,
            "image_id": image_id,
            "member_name": member_name,
            "member_info" : member_info
        }
    


    def _filter_tar_members(self, member_dict, stem):

        inner_path = f"{stem}/DATA/"

        return member_dict["member_name"].startswith(inner_path) \
            and member_dict["suffix"] in ["IMG","LBL"] \
            and member_dict["image_type"] == "GEOMED"



    def _get_members_info_file(self, tar, stem ):

        tar_file_info_file = f"{self.cache_path}/member_info/memberinfo_{stem}.p"

        if not os.path.isfile( tar_file_info_file ):
        
            all_tar_file_members = tar.getmembers()
        
            with open(tar_file_info_file, 'wb') as fp:
                pickle.dump(all_tar_file_members, fp)

        else:
            with open(tar_file_info_file, 'rb') as fp:
                all_tar_file_members = pickle.load(fp)      

        return all_tar_file_members


    def _get_cache_file_list(self) -> dict:

        import glob

        cache_file_list = glob.glob( str( self.cache_path / "tar_members" / "**" ), recursive=True)

        if len(cache_file_list) == 0:
            os.makedirs( self.cache_path / "tar_members", exist_ok=True)

        return { g:True for g in cache_file_list }



    def process(self, item ):
        import tarfile

        # print( item )

        tar_file_path = item["tar_file_path"]
        tar_stem = item["stem"]


        
        with tarfile.open( tar_file_path, "r") as tar:
            all_tar_file_members = self._get_members_info_file( tar, tar_stem )

            parsed_member_info = [ self._parse_member_name(mi) for mi in all_tar_file_members ]

            logger.info( f"--- working on tar file {tar_file_path} with length {len(parsed_member_info)} ---" )

            for n, member_dict in enumerate( tqdm.tqdm( parsed_member_info) ):

                if self._filter_tar_members(member_dict,tar_stem):

                    local_full_path = self.cache_path / "tar_members"

                    # if not os.path.isfile( local_full_path ):
                    
                    local_full_fn = local_full_path / member_dict["member_info"].get_info()["name"]

                    # print( self.cached_files_list )
                    # print( local_full_fn)
                    # print( not local_full_fn in self.cached_files_list )

                    if not str( local_full_fn ) in self.cached_files_list:
                        # print("extractnig")
                        # raise Exception("stop")
                        tar.extract( member_dict["member_info"], path=local_full_path )  
                        pass

                    member_dict["local_file_path"] = local_full_fn

                    yield member_dict



class LoadAndStoreMetadata(pipeline.Task):

    def __init__(self, name):

        super().__init__( name )

    def process(self, item ):
        
        fn = item["local_file_path"]

        # print( fn )
    
        with open( fn ) as fp:
            metadata = pvl.loads( fp.read() )

            flattened = flatten_vicar_object(metadata, to_exclude=["^VICAR_HEADER", "^IMAGE", "SOURCE_PRODUCT_ID"])

            # create_user_from_dict(db.SessionLocal() , d=flattened)
            with db.SessionLocal() as session:
                upsert_foobar(session, voyager_image_dict=flattened)

        return item




# class ExtractTarMembers(pipeline.Task):

#     def __init__(self, name, cache_path):

#         super().__init__( name )

#         self.cache_path = cache_path


#     def should_extract( self, member_info ) -> bool:
#         pass



#     def process(self, item ):
#         import tarfile

#         all_tar_file_members = item["all_tar_file_members"]
#         stem = item["stem"]
#         tar_file_info_file = item["tar_file_info_file"]

        


# class ExtractMetadata(pipeline.Task):

#     def __init__(self, name, cache_path):

#         super().__init__( name )

#         self.cache_path = cache_path

#         self.cached_files_list = self._get_cache_file_list()
#         logger.info( f"cached files list has {len(self.cached_files_list)} entries" )



#                             if inner_suffix == "LBL":

#                                 with open( local_full_path ) as fp:
#                                     metadata = pvl.loads( fp.read() )
        
#                                 this_metadata = { **r, 
#                                     **{
#                                         "image_time": metadata["IMAGE_TIME"],
#                                         "filter_name": metadata["FILTER_NAME"],
#                                         "filter_number": metadata["FILTER_NUMBER"],
#                                         "instrument_name": metadata["INSTRUMENT_NAME"],
#                                         "gain_mode_id": metadata["GAIN_MODE_ID"],
#                                         "target_name": metadata["TARGET_NAME"],
#                                         "exposure_duration": metadata["EXPOSURE_DURATION"],
#                                     }
#                                 }
                                
#                                 metadata_lbl.append( this_metadata )       






# metadata_lbl = []
# metadata_img = []

#         if len( circles ) == 0:
#             circles = [[[500, 500, 5 ]]]
# m = 0

# all_tar_files = glob.glob(f"{data_path}VGISS_61*.tar.gz")

# for tar_file_cnt, g in enumerate( all_tar_files ):

#     print(f"loading file {tar_file_cnt}/{len(all_tar_files)}")
    
#     p = pathlib.Path( g )

#     stem = p.stem.replace(".tar","")

#     print( f"{g} / {stem}" )

#     with tarfile.open( g, "r") as tar:

#         # only re-read the tar index (time consuming) if we must
        
#         tar_file_info_file = f"{local_landing_path}memberinfo_{stem}.p"

#         if not os.path.isfile( tar_file_info_file ):
#             all_tar_file_members = tar.getmembers()
        
#             with open(tar_file_info_file, 'wb') as fp:
#                 pickle.dump(all_tar_file_members, fp)

#         else:
#             with open(tar_file_info_file, 'rb') as fp:
#                 all_tar_file_members = pickle.load(fp)      

            
        
#         for n,member_info in enumerate( tqdm.tqdm( all_tar_file_members, position=0, leave=True) ):

#             member_name = member_info.get_info()["name"]
            
#             inner_path = f"{stem}/DATA/"
            
#             if member_name.startswith( inner_path ):

#                 inner_p = pathlib.Path( member_name )
            
#                 inner_suffix = inner_p.suffix.replace(".","")
#                 inner_stem = inner_p.stem
            
#                 if not member_info.isdir():
                    
#                     if "_" not in inner_stem:
#                         raise Exception(f"found non-conforming filename : {stem} " )
                
#                     else:

#                         if "GEOMED" in inner_stem and inner_suffix in ["IMG","LBL"]:
                            
#                             id = inner_stem.split("_")[0]
#                             the_type = inner_stem.split("_")[1]
                        
#                             r = {
#                                 "stem": inner_stem,
#                                 "type": the_type,
#                                 "suffix": inner_suffix,
#                                 "id": id,
#                                 "fn": member_name,
#                                 "tar_fn": g,
#                             }

#                             # extract the file if necessary
#                             local_full_path = f"{local_landing_path}{member_name}"

#                             if not os.path.isfile( local_full_path ):
#                                 tar.extract( member_info, path=local_landing_path )  
                            

#                             if inner_suffix == "LBL":

#                                 with open( local_full_path ) as fp:
#                                     metadata = pvl.loads( fp.read() )
        
#                                 this_metadata = { **r, 
#                                     **{
#                                         "image_time": metadata["IMAGE_TIME"],
#                                         "filter_name": metadata["FILTER_NAME"],
#                                         "filter_number": metadata["FILTER_NUMBER"],
#                                         "instrument_name": metadata["INSTRUMENT_NAME"],
#                                         "gain_mode_id": metadata["GAIN_MODE_ID"],
#                                         "target_name": metadata["TARGET_NAME"],
#                                         "exposure_duration": metadata["EXPOSURE_DURATION"],
#                                     }
#                                 }
                                
#                                 metadata_lbl.append( this_metadata )                                    

#                             elif inner_suffix == "IMG":
                            
#                                 v_im = vicar.VicarImage( local_full_path )

#                                 fn_stem = member_name.replace(".IMG","").replace("/","_")
                                
#                                 out_jpg_fn = f"{output_images}jpg/{fn_stem}.jpg"
#                                 out_p_fn = f"{output_images}pickle/{fn_stem}.p"

#                                 norm_and_save_grey_image( v_im.array, out_jpg_fn )
                                
#                                 # normed_arr = normalize_clip( v_im.array, in_min=v_im.array.min() , in_max=v_im.array.max() )

#                                 # dim1 = normed_arr.shape[1]
#                                 # dim2 = normed_arr.shape[2]

#                                 metadata_img.append( r )
                                
#                                 # plt.imsave( out_jpg_fn,
#                                 #     np.stack([normed_arr]*3, axis=-1).reshape(dim1,dim2,3),
#                                 #     cmap="grey"
#                                 #           )

#                                 with open( out_p_fn, "wb" ) as fp:
#                                     pickle.dump(v_im.array, fp)
                                    
                            
#                             else:
#                                 raise Exception(f"weird file with no known suffix, {r}")    
#     break                            





def main():

    # p = pipeline.Pipeline()

    list_files_result = list( ListTarFiles( name="list_tars", source_path=config.source_path ).process([1]) )

    extract_tar_file_obj = ExtractTarfile( name="list_tar_members", cache_path=config.cache_path )

    extract_files_result = list( itertools.chain.from_iterable([ list(extract_tar_file_obj.process(item)) for item in list_files_result ]) )

    lbl_list = [ item for item in extract_files_result if item["member_name"].endswith(".LBL") ]
    img_list = [ item for item in extract_files_result if item["member_name"].endswith(".IMG") ]

    # print( lbl_list[:50] )

    logger.info( f"Found {len(lbl_list)} LBL files and {len(img_list)} IMG files, now INSERTing the LBL files to sqllite." )

    if os.path.exists( config.db_loaded_fn ):
        logger.info( f"Database already loaded, skipping insertion." )
    else:

        load_and_store_metadata_obj = LoadAndStoreMetadata( name="load_and_store_metadata" )

        loaded_lbl_list = [ load_and_store_metadata_obj.process(item) for item in tqdm.tqdm(lbl_list) ]

        # print( len( loaded_lbl_list))

        open( config.db_loaded_fn, 'w').close()



if __name__ == "__main__":
    main()

    # fn = "/home/lobdellb/repos/voyager_flyby/cache/tar_members/VGISS_6115/DATA/C35170XX/C3517001_GEOMED.LBL"

    # with open( fn ) as fp:
    #     metadata = pvl.loads( fp.read() )

    #     flattened = flatten_vicar_object(metadata, to_exclude=["^VICAR_HEADER", "^IMAGE", "SOURCE_PRODUCT_ID"])

    #     create_user_from_dict(db.SessionLocal() , d=flattened)


    print("Done")